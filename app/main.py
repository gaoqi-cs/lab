import multiprocessing
#多线程
multiprocessing.freeze_support() #线程
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gc
import uvicorn
from routers import test,test2,train,chat
from utils.common_utils import default_logger
from config.config import settings
from services.classify_service import load_resnet50_from_local_safetensors
from services.classify_service import ClassifyService
from pathlib import Path
import asyncio
from services.train_service import train_service
from routers import classify, knowledge, inspections

app=FastAPI(
    title="校园实验室安全巡检助手",
    version="1.0.0",
    description="用于实验室安全巡检、场景模型训练、规程资料检索与应急问答的 API 接口",
    
    
)
#跨域配置
app.add_middleware(
                   CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )

@app.on_event("startup")
async def startup_event():
    #启动时的初始化操作
    #load_resnet50_from_local_safetensors()
    #ClassifyService().fintune(Path(settings.UPLOAD_DATASET_UNZIPED_DIR))
    train_service.set_main_loop(asyncio.get_running_loop())
    default_logger.info("应用启动完成")
@app.on_event("shutdown")
async def shutdown_event():
    #关闭时清理操作
    gc.collect() #强制清除当前程序占用的内存和显存
    default_logger.info("应用关闭")
    
@app.get("/health")
async def health():
    return {"status": "ok", "title": "校园实验室安全巡检助手"}

#注册路由
#app.include_router(test.router)
#app.include_router(test2.router)
app.include_router(train.router)
app.include_router(classify.router)
app.include_router(knowledge.router)
app.include_router(inspections.router)
app.include_router(chat.router)
if __name__ == "__main__":
    import uvicorn
    # 启动服务器0
    uvicorn.run("main:app",host="0.0.0.0",port=8001,reload=True)
