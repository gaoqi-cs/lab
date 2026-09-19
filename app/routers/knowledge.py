from fastapi import APIRouter, File, Query, UploadFile

from config.config import settings
from pathlib import Path
from fastapi import HTTPException
from utils.common_utils import default_logger
import uuid
import json
import threading
from services.rag_service import rag_service
router = APIRouter(prefix="/api/knowledge")
catalog_lock = threading.Lock()


def load_catalog():
    catalog_path = Path(settings.KNOWLEDGE_CATALOG_FILE)
    if not catalog_path.exists():
        return []
    try:
        items = json.loads(catalog_path.read_text(encoding="utf-8"))
        return items if isinstance(items, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_catalog(items):
    catalog_path = Path(settings.KNOWLEDGE_CATALOG_FILE)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


@router.get("/documents")
async def list_documents(query: str = Query(default="", max_length=100)):
    keyword = query.strip().lower()
    with catalog_lock:
        documents = load_catalog()
    if keyword:
        documents = [item for item in documents if keyword in item["name"].lower()]
    return {"documents": documents}

@router.post("/upload")
async def upload_document(
    
    file:UploadFile = File(...)
   
):
    file_name = Path(file.filename or "").name
    if not file_name:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    #如果没有则"",反之
    ext = Path(file_name).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        default_logger.error(f"文件扩展名错误:{ext}")
        raise HTTPException(status_code=400,detail="文件扩展名不支持")
    temp_dir = Path(settings.UPLOAD_FILE_DIR)
    unique_name = f"{uuid.uuid4().hex}_{file_name}"
    #保存文件的路径
    save_path=temp_dir / unique_name
    
    try:
        content = await file.read()
        if len(content) > settings.MAX_UPLOAD_FILE_SIZE:
            default_logger.error(f"文件大小超过最大限制:{settings.MAX_UPLOAD_FILE_SIZE}")
            raise HTTPException(status_code=400,detail="文件大小超过最大限制")
        #保存文件
        with open(save_path,"wb") as f:
            f.write(content)
        default_logger.info(f"保存文件:{save_path}")
        
        result = rag_service.add_document(str(save_path))
        if not result["success"]:
            default_logger.error(f"添加文件到RAG失败:{result['message']}")
            raise HTTPException(status_code=500,detail="添加文件到RAG失败")
        document = {
            "id": uuid.uuid4().hex,
            "name": file_name,
            "type": ext.lstrip(".").upper(),
            "desc": f"已入库 {result['trunk_count']} 个文本片段"
        }
        with catalog_lock:
            catalog = [item for item in load_catalog() if item["name"] != file_name]
            catalog.insert(0, document)
            save_catalog(catalog)
        return {"message":"文件上传成功","success":True,"document":document}
    except HTTPException:
        raise
    except Exception as e:
        default_logger.error(f"文件上传失败:{e}")
        raise HTTPException(status_code=500,detail="文件上传失败")
