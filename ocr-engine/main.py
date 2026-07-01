import logging
import os
from typing import Optional

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr-engine")

IS_DEV = os.getenv("APP_ENV", "production").lower() == "development"
MINIO_SERVICE_URL = os.getenv("MINIO_SERVICE_URL", "http://minio-service:8001")

# ── Response models (drives Swagger schema) ──────────────────────────────────

class UploadProxyResponse(BaseModel):
    file_uuid: str = Field(..., description="UUID identifying the stored image")
    raw_key: str = Field(..., description="MinIO key for the raw upload")
    processed_key: Optional[str] = Field(None, description="MinIO key for the preprocessed image (null if preprocessing failed)")


class ExtractionResult(BaseModel):
    license_plate:           Optional[str]   = Field(None, description="(A) Registration plate number")
    first_registration_date: Optional[str]   = Field(None, description="(B) First registration date — YYYY-MM-DD")
    vin:                     Optional[str]   = Field(None, description="(E) Vehicle Identification Number — 17 chars")
    owner_name:              Optional[str]   = Field(None, description="(C.1) Titular owner name and surname")
    registration_address:    Optional[str]   = Field(None, description="(C.3) Registration address of the vehicle")
    brand:                   Optional[str]   = Field(None, description="(D.1) Vehicle make / brand")
    type_variant_version:    Optional[str]   = Field(None, description="(D.2) Type, variant and version code")
    cnit_code:               Optional[str]   = Field(None, description="(D.2.1) National Type Identification Code")
    commercial_name:         Optional[str]   = Field(None, description="(D.3) Commercial model name (e.g. Clio, 308)")
    gross_vehicle_weight_kg: Optional[int]   = Field(None, description="(F.1) Gross Vehicle Weight / MMA in kg")
    adjusted_gvw_kg:         Optional[int]   = Field(None, description="(F.2) Adjusted GVW in kg")
    max_towing_weight_kg:    Optional[int]   = Field(None, description="(F.3) Gross Combination Weight Rating in kg")
    max_trailer_weight_kg:   Optional[int]   = Field(None, description="(O.1) Maximum trailer weight in kg")
    eu_category:             Optional[str]   = Field(None, description="(J.1) EU vehicle category — M1, N1, L3e…")
    eu_bodywork_code:        Optional[str]   = Field(None, description="(J.2) EU bodywork code — AA, AB, AC, AG…")
    national_bodywork:       Optional[str]   = Field(None, description="(J.3) National bodywork designation")
    type_approval_number:    Optional[str]   = Field(None, description="(K) Type approval number — e.g. e2*2007/46*0123")
    engine_capacity_cm3:     Optional[int]   = Field(None, description="(P.1) Engine displacement in cm³")
    max_power_kw:            Optional[float] = Field(None, description="(P.2) Maximum net power in kW")
    fuel_type:               Optional[str]   = Field(None, description="(P.3) Fuel type — Diesel, Petrol, Electric, Hybrid…")
    fiscal_power_cv:         Optional[int]   = Field(None, description="(P.6) Fiscal / administrative horsepower in CV")
    seating_capacity:        Optional[int]   = Field(None, description="(S.1) Total seats including driver")
    standing_places:         Optional[int]   = Field(None, description="(S.2) Standing places (0 for private cars)")
    co2_emissions_g_km:      Optional[int]   = Field(None, description="(V.7) CO2 emissions in g/km")
    euro_class:              Optional[str]   = Field(None, description="(V.9) Euro emission standard — Euro 5, Euro 6d…")
    confidence:              str             = Field(...,  description="Extraction quality: high | low | failed")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OCR Engine",
    description="""
## Belgian Carte Grise — OCR Engine

Extracts structured data from Belgian Vehicle Registration Certificates using a **Hybrid Pipeline**:

1. **PaddleOCR** — local text extraction from mobile photos (free, open-source)
2. **GPT-4o-mini** — contextual structuring, typo correction, and EU code mapping

### Typical flow
```
POST /upload          →  returns file_uuid
POST /process/{uuid}  →  returns ExtractionResult JSON
```

> **Swagger UI** is only visible when `APP_ENV=development` in `.env`.
""",
    version="1.0.0",
    contact={"name": "OCR Marketplace", "email": "nadirsatori31@gmail.com"},
    docs_url="/docs"      if IS_DEV else None,
    redoc_url="/redoc"    if IS_DEV else None,
    openapi_url="/openapi.json" if IS_DEV else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

from services.llm import structure_extraction   # noqa: E402
from services.ocr import extract_text           # noqa: E402
from services.storage import fetch_best_image   # noqa: E402


@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    response_description="Service is up",
)
def health():
    return {"status": "ok"}


@app.post(
    "/upload",
    response_model=UploadProxyResponse,
    tags=["Storage (proxy)"],
    summary="Upload a vehicle registration image",
    description="Proxies the upload to the MinIO Storage Service. Returns a `file_uuid` to use with `/process/{file_uuid}`.",
    responses={
        200: {"content": {"application/json": {"example": {
            "file_uuid": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "raw_key": "raw/3fa85f64-5717-4562-b3fc-2c963f66afa6.jpg",
            "processed_key": "processed/3fa85f64-5717-4562-b3fc-2c963f66afa6.jpg",
        }}}},
        400: {"description": "Unsupported file type or empty file"},
        413: {"description": "File exceeds 10 MB limit"},
        503: {"description": "MinIO storage service unreachable"},
    },
)
async def upload_proxy(file: UploadFile = File(..., description="JPEG, PNG or PDF — max 10 MB")):
    raw = await file.read()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MINIO_SERVICE_URL}/upload",
                files={"file": (file.filename, raw, file.content_type)},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"MinIO service unreachable: {exc}")


@app.post(
    "/process/{file_uuid}",
    response_model=ExtractionResult,
    tags=["OCR"],
    summary="Extract data from an uploaded image",
    description="""
Runs the full **PaddleOCR → GPT-4o-mini** pipeline on the image identified by `file_uuid`.

- Fetches the preprocessed image from MinIO (falls back to raw if preprocessing failed)
- Extracts raw text blocks with PaddleOCR (`lang=latin` for FR/NL/DE)
- Sends the raw text to GPT-4o-mini with a prompt that maps all EU field codes (A → V.9)
- Returns a normalized JSON with 25 fields + `confidence`
""",
    responses={
        200: {"content": {"application/json": {"example": {
            "license_plate": "1-ABC-234",
            "first_registration_date": "2019-05-14",
            "vin": "WAUZZZ8K79A123456",
            "brand": "Peugeot",
            "commercial_name": "308",
            "engine_capacity_cm3": 1997,
            "max_power_kw": 110,
            "fuel_type": "Diesel",
            "fiscal_power_cv": 6,
            "euro_class": "Euro 6d",
            "co2_emissions_g_km": 142,
            "seating_capacity": 5,
            "eu_category": "M1",
            "eu_bodywork_code": "AB",
            "confidence": "high",
        }}}},
        404: {"description": "No image found for the given file_uuid"},
        422: {"description": "No text detected in the image"},
        502: {"description": "PaddleOCR or GPT-4o-mini call failed"},
    },
)
def process(file_uuid: str):
    try:
        image_bytes = fetch_best_image(file_uuid)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No image found for file_uuid={file_uuid}")

    try:
        raw_text = extract_text(image_bytes)
    except Exception as exc:
        logger.exception("PaddleOCR extraction failed for %s", file_uuid)
        raise HTTPException(status_code=502, detail=f"OCR extraction failed: {exc}")

    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="No text detected in image")

    try:
        structured = structure_extraction(raw_text)
    except Exception as exc:
        logger.exception("LLM structuring failed for %s", file_uuid)
        raise HTTPException(status_code=502, detail=f"LLM structuring failed: {exc}")

    return structured
