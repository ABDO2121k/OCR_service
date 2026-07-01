from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    file_uuid: str
    raw_key: str
    processed_key: Optional[str]


class ExtractionResult(BaseModel):
    license_plate: Optional[str]
    vin: Optional[str]
    engine_capacity: Optional[str]
    first_registration: Optional[str]
    confidence: str  # "high" | "low" | "failed"
