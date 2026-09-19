from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import asyncio
from services.train_service import train_service
from utils.common_utils import default_logger
from config.config import settings
from fastapi import HTTPException, Request
from pathlib import Path
import shutil
import base64
import io
import re
from PIL import Image

class ConnectionManager:
    #websocket管理类
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        
    async def connect(self,websocket:WebSocket):
        await websocket.accept()
        #如果连接成功，则把ws添加到active_connections
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket:WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
    async def broadcast(self,message:dict):
        #广播消息到所有来凝结的客户端
        if not self.active_connections:
            return
        # 并播消息到所有连接的客户端
        
        tasks = [conn.send_json(message) for conn in self.active_connections]
        #等待所有任务完成
        await asyncio.gather(*tasks,return_exceptions=True)
        
manager = ConnectionManager()

def broadcast_status(status: dict):
    loop = train_service._main_loop
    if loop is None or not loop.is_running():
        return
    asyncio.run_coroutine_threadsafe(manager.broadcast(status),loop)
    
train_service.set_broadcast_callback(broadcast_status)

router = APIRouter(prefix="/api/train")

 #ws端点
@router.websocket("/ws")
async def websocket_endpoint(websocket:WebSocket):
    #建立连接
    await manager.connect(websocket)
    try:
        #发送当前状态 
        await websocket.send_json(train_service.get_status())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        #如果客户要断链，则从active_connections中移除ws
        manager.disconnect(websocket)
    except Exception as e:
        default_logger.error(f"处理ws消息失败:{e}")
        manager.disconnect(websocket)

@router.get("/status")
async def get_train_status():
    return train_service.get_status()


@router.get("/dataset")
async def get_dataset_summary():
    """返回当前训练数据集中的类别和图片数量。"""
    train_dir = Path(settings.UPLOAD_DATASET_UNZIPED_DIR)
    classes = []
    for class_dir in sorted(train_dir.iterdir(), key=lambda item: item.name):
        if not class_dir.is_dir():
            continue
        image_count = sum(
            1 for file in class_dir.iterdir() if file.is_file()
        )
        classes.append({"name": class_dir.name, "count": image_count})
    return {"classes": classes, "total": sum(item["count"] for item in classes)}


@router.post("/dataset/delete")
async def delete_dataset_class(request: Request):
    """删除一个尚未参与训练的完整案例及其样本图片。"""
    if train_service.status == settings.TRAIN_STATUS_RUNNING:
        raise HTTPException(status_code=409, detail="训练进行中，无法删除训练案例")

    body = await request.json()
    label = body.get("label")
    if not isinstance(label, str) or not label.strip():
        raise HTTPException(status_code=400, detail="请选择要删除的训练案例")
    label = label.strip()

    train_dir = Path(settings.UPLOAD_DATASET_UNZIPED_DIR).resolve()
    class_dir = (train_dir / label).resolve()
    if class_dir.parent != train_dir or not class_dir.is_dir():
        raise HTTPException(status_code=404, detail="训练案例不存在或名称无效")

    shutil.rmtree(class_dir)
    return {"message": f"已删除训练案例：{label}"}


@router.post("/start")
async def start_training():
    """在用户确认样本收集完成后，显式启动训练。"""
    train_dir = Path(settings.UPLOAD_DATASET_UNZIPED_DIR)
    class_counts = {
        item.name: sum(1 for file in item.iterdir() if file.is_file())
        for item in train_dir.iterdir()
        if item.is_dir()
    }
    insufficient = [name for name, count in class_counts.items() if count < 2]
    if len(class_counts) < 2 or insufficient:
        details = "、".join(insufficient) if insufficient else "至少两个类别"
        raise HTTPException(
            status_code=400,
            detail=f"训练至少需要两个类别，且每类至少 2 张有效图片；当前需补充：{details}",
        )

    if not train_service.start_trainning(train_dir, settings.FULL_EPOCHS):
        raise HTTPException(status_code=409, detail="正在运行中或启动训练失败")
    return {
        "message": "训练已启动",
        "status": settings.TRAIN_STATUS_RUNNING,
        "class_counts": class_counts,
    }

@router.post("/stop")
async def stop_training():
    if train_service.status != settings.TRAIN_STATUS_RUNNING:
        raise HTTPException(status_code=400, detail="训练未运行中")
    success = train_service.request_stop()
    if not success:
        raise HTTPException(status_code=400,detail="终止训练失败")
    return {"message":"训练已终止"}

@router.post("/finetune")
async def finetune_resnet50(request: Request):
    content_type = request.headers.get("content-type","")
    if content_type.startswith("application/json"):
        #从请求体中获取json字符串
        body = await request.json()
        label = body.get("label")
        clear_old = body.get("clearOld",False)
        default_logger.info(f"是否清除旧模型:{clear_old}")
        images_base64 = body.get("images",[])
        
        if not images_base64:
            raise HTTPException(status_code=400,detail="请上传图片")
        
        if not isinstance(label, str) or not label.strip():
            raise HTTPException(status_code=400,detail="请上传标签")
        label = label.strip()
        if len(label) > 50 or not re.fullmatch(r"[\w\-\u4e00-\u9fff ]+", label):
            raise HTTPException(status_code=400, detail="标签仅支持中英文、数字、空格、下划线和短横线")
        
        default_logger.info(f"开始微调ResNet50,标签：{label},是否清楚旧模型:{clear_old}")
        
        
        train_dir = Path(settings.UPLOAD_DATASET_UNZIPED_DIR)
        if clear_old:
            default_logger.info("清楚旧数据")
            for item in train_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
        #新加入的分类目录
        class_dir = train_dir / label
        class_dir.mkdir(parents=True,exist_ok=True)
        #实际保存的图片数量
        saved_count = 0
        existing_count = sum(1 for file in class_dir.iterdir() if file.is_file())
        for idx, b64_data in enumerate(images_base64):
            try:
                if "," in b64_data:
                    b64_data = b64_data.split(",", 1)[-1]
                # Base64 字符串长度必须是 4 的倍数。
                b64_data += "=" * (4 - len(b64_data) % 4) if len(b64_data) % 4 else ""
                img_bytes = base64.b64decode(b64_data, validate=True)
                img = Image.open(io.BytesIO(img_bytes))
                image_format = img.format or "JPEG"
                img.verify()
                ext = {
                    "PNG": ".png",
                    "GIF": ".gif",
                    "WEBP": ".webp",
                }.get(image_format, ".jpg")

                # 使用已有数量偏移，避免后续批次覆盖前面的图片。
                safe_name = f"{label}_{existing_count + idx + 1}{ext}"
                file_path = class_dir / safe_name
                with open(file_path, "wb") as image_file:
                    image_file.write(img_bytes)
                saved_count += 1
                    
            except Exception as e:
                default_logger.error(f"保存图片失败:{e}")
        
        if saved_count == 0:
            raise HTTPException(status_code=400,detail="保存的图片数量为0")
          
        default_logger.info(f"成功保存{saved_count}张图片到目录:{class_dir}")
        class_counts = {
            item.name: sum(1 for file in item.iterdir() if file.is_file())
            for item in train_dir.iterdir()
            if item.is_dir()
        }
        return JSONResponse(
            status_code=202,
            content={
                "message": "本批样本已保存，可继续添加其他批次；准备完成后再点击开始训练。",
                "status": "samples_saved",
                "saved_count": saved_count,
                "class_counts": class_counts,
            },
        )
    else:
        raise HTTPException(status_code=415,detail="请上传json格式的请求体")
        
            
