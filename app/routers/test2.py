
from fastapi import APIRouter

#实例化路优器

router = APIRouter(prefix="/api/test2")

@router.get("/xxx")

async def test2():
    #返回json格式的数据

    return{"隔壁":"水哥"}