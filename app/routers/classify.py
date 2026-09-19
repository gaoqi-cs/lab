from fastapi import APIRouter,HTTPException
import base64
from services.classify_service import classify_service
from utils.common_utils import default_logger
from fastapi import Request
router = APIRouter(prefix="/api/classify")


def decode_base64_image(base64_str:str)->bytes:
    
    
    if base64_str.startswith("data:image"):
        #前端传回的base64字符串，有且只有一个，所以只需要搜索并分割1次即可
        base64_str = base64_str.split(",",1)[-1]
    try:
        return base64.b64decode(base64_str)
    except Exception as e:
        default_logger.error(f"base64字符串解码失败: {e}")
        raise ValueError(f"base64字符串解码失败: {e}")
@router.post("/")
async def classify_image(request:Request):
    content_type = request.headers.get("content-type", "")
    try: 
        if content_type.startswith("application/json"):
            body = await request.json()
            image_data = body.get("image")
            if not image_data:
                default_logger.error("未上传图片")
                raise HTTPException(status_code=400,detail=" 请上传图片")
        
            img_bytes = decode_base64_image(image_data)
            result = classify_service.predict(img_bytes)
            return {"result":result}
        else:
            default_logger.error("不支持的content-type")
            raise HTTPException(status_code=400,detail="不支持的content-type")
    except HTTPException:
        #如果捕获到HTTPException，则直接抛出
        raise
    except Exception as e:
        default_logger.error(f"分类图片失败: {e}")
        raise HTTPException(status_code=500,detail="分类图片失败")
