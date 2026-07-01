import io
import os
import uuid
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from minio import Minio
from pydantic import BaseModel, Field

from utils.preprocess import preprocess_image

IS_DEV = os.getenv("APP_ENV", "production").lower() == "development"

# ── Response model ────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    file_uuid: str = Field(..., description="Unique UUID identifying the stored file — pass to /process/{file_uuid}")
    raw_key: str = Field(..., description="MinIO object key for the raw uploaded file")
    processed_key: Optional[str] = Field(None, description="MinIO key for the preprocessed image (None if preprocessing failed)")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MinIO Storage Service",
    description="""
## MinIO Storage Service

Handles secure upload and preprocessing of Belgian Carte Grise images.

### Upload pipeline
1. Client sends a JPEG / PNG / PDF via `POST /upload` (max 10 MB)
2. The raw file is stored in MinIO under `raw/{uuid}.ext`
3. The image is preprocessed — **deskew** (≤ 15°), **CLAHE contrast normalization**, **resize to 2000 px width**
4. The processed image is stored under `processed/{uuid}.jpg`
5. The `file_uuid` is returned — pass it to the **OCR Engine** (`POST /process/{file_uuid}`)

> **Swagger UI** is only visible when `APP_ENV=development` in `.env`.
""",
    version="1.0.0",
    contact={"name": "OCR Marketplace", "email": "nadirsatori31@gmail.com"},
    docs_url="/docs"           if IS_DEV else None,
    redoc_url="/redoc"         if IS_DEV else None,
    openapi_url="/openapi.json" if IS_DEV else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MinIO client ──────────────────────────────────────────────────────────────

_raw_endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
_endpoint = _raw_endpoint.replace("http://", "").replace("https://", "")

minio_client = Minio(
    _endpoint,
    access_key=os.getenv("MINIO_ROOT_USER", "admin"),
    secret_key=os.getenv("MINIO_ROOT_PASSWORD", "password123"),
    secure=_raw_endpoint.startswith("https://"),
)

BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "vehicle-registrations")
ALLOWED_TYPES = {"image/jpeg", "image/png", "application/pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# ── Routes ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def ensure_bucket():
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)


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
    response_model=UploadResponse,
    tags=["Storage"],
    summary="Upload a vehicle registration image",
    description="Accepts a JPEG, PNG, or PDF file (max 10 MB). Stores the raw file and a preprocessed version in MinIO. Returns a `file_uuid` to pass to the OCR Engine.",
    responses={
        200: {"content": {"application/json": {"example": {
            "file_uuid": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "raw_key": "raw/3fa85f64-5717-4562-b3fc-2c963f66afa6.jpg",
            "processed_key": "processed/3fa85f64-5717-4562-b3fc-2c963f66afa6.jpg",
        }}}},
        400: {"description": "Unsupported file type or empty file"},
        413: {"description": "File exceeds 10 MB limit"},
    },
)
async def upload(file: UploadFile = File(..., description="Vehicle registration image — JPEG, PNG, or PDF, max 10 MB")):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB limit")

    file_uuid = str(uuid.uuid4())
    extension = file.filename.rsplit(".", 1)[-1].lower() if file.filename and "." in file.filename else "jpg"
    raw_key = f"raw/{file_uuid}.{extension}"
    processed_key = f"processed/{file_uuid}.jpg"

    minio_client.put_object(
        BUCKET_NAME, raw_key, io.BytesIO(raw_bytes),
        length=len(raw_bytes), content_type=file.content_type,
    )

    try:
        processed_bytes = preprocess_image(raw_bytes)
        minio_client.put_object(
            BUCKET_NAME, processed_key, io.BytesIO(processed_bytes),
            length=len(processed_bytes), content_type="image/jpeg",
        )
    except Exception:
        processed_key = None

    return JSONResponse({
        "file_uuid": file_uuid,
        "raw_key": raw_key,
        "processed_key": processed_key,
    })
