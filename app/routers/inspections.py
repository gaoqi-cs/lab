from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.inspection_service import inspection_service


router = APIRouter(prefix="/api/inspections")


class InspectionRecordRequest(BaseModel):
    image_name: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=120)
    confidence: str = Field(min_length=1, max_length=32)


@router.post("/", status_code=201)
async def create_inspection_record(request: InspectionRecordRequest):
    image_name = request.image_name.strip()
    label = request.label.strip()
    confidence = request.confidence.strip()
    if not image_name or not label or not confidence:
        raise HTTPException(status_code=400, detail="巡检记录字段不能为空")
    try:
        record = inspection_service.create(image_name, label, confidence)
        return {"success": True, "record": record}
    except OSError as error:
        raise HTTPException(status_code=500, detail="巡检记录保存失败") from error


@router.get("/")
async def list_inspection_records(limit: int = Query(default=5, ge=1, le=50)):
    return {"records": inspection_service.list_recent(limit)}
