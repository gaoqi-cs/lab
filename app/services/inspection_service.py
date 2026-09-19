import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config.config import settings


class InspectionService:
    def __init__(self):
        self._lock = threading.Lock()
        self._file_path = Path(settings.INSPECTION_RECORD_FILE)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_records(self):
        if not self._file_path.exists():
            return []
        try:
            data = json.loads(self._file_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def create(self, image_name: str, label: str, confidence: str):
        record = {
            "id": uuid.uuid4().hex,
            "image_name": image_name[:120],
            "label": label[:120],
            "confidence": confidence[:32],
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        }
        with self._lock:
            records = self._read_records()
            records.insert(0, record)
            self._file_path.write_text(
                json.dumps(records[:1000], ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return record

    def list_recent(self, limit: int):
        with self._lock:
            return self._read_records()[:limit]


inspection_service = InspectionService()
