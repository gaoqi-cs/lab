import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from services.llm_service import llm_service
from services.rag_service import rag_service
from utils.common_utils import default_logger


router = APIRouter(prefix="/api/chat")
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=64, le=4096)


async def generate(request:ChatRequest):
    full_response = ""
    try:
        result = rag_service.query(request.message)
        sources = result.get("sources") or result.get("source") or []
        answer = result.get("answer","")

        # 只有本地资料没有命中时，才调用通用模型兜底。
        if result.get("use_knowledge") and answer:
            full_response = answer
            if sources:
                source_names = [Path(source).name for source in sources[:10]]
                full_response += f"\n来源文件:{','.join(source_names)}"
        else:
            default_logger.info("知识库未命中，调用通用大模型兜底")
            resp = llm_service.generate(
                user_prompt=request.message,
                system_prompt=(
                    "你是校园实验室安全巡检助手。请以谨慎、清晰的方式回答实验室"
                    "安全操作问题；不确定时说明需要查阅本校正式规程或联系实验室管理员。"
                ),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            full_response = resp
            full_response += "\n> 回答来自通用安全助手，未命中本地安全规程资料。"

        yield f"data:{json.dumps({'chunk': full_response}, ensure_ascii=False)}\n\n"
        default_logger.info(f"生成完成:{full_response}")
        yield f"data:{json.dumps({'chunk': '', 'done': True}, ensure_ascii=False)}\n\n"
    except Exception as e:
        default_logger.error(f"问答失败:{e}")
        yield f"data:{json.dumps({'error': str(e), 'done': True}, ensure_ascii=False)}\n\n"

@router.post("/stream")
async def chat_stream(request:ChatRequest):
    default_logger.info(f"开始处理:{request.message}")
    return StreamingResponse(
        generate(request),
        media_type="text/event-stream",
        headers={
            "Connection": "keep-alive",
            "Cache-Control": "no-cache",
        },
    )
