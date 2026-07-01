import os

from minio import Minio

_raw_endpoint = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
_endpoint = _raw_endpoint.replace("http://", "").replace("https://", "")

minio_client = Minio(
    _endpoint,
    access_key=os.getenv("MINIO_ROOT_USER", "admin"),
    secret_key=os.getenv("MINIO_ROOT_PASSWORD", "password123"),
    secure=_raw_endpoint.startswith("https://"),
)

BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME", "vehicle-registrations")


def _find_object_key(file_uuid: str, prefix: str) -> str | None:
    objects = minio_client.list_objects(BUCKET_NAME, prefix=f"{prefix}/{file_uuid}")
    for obj in objects:
        return obj.object_name
    return None


def fetch_best_image(file_uuid: str) -> bytes:
    """Fetches the preprocessed image for OCR, falling back to the raw
    upload if preprocessing failed or was skipped."""
    key = _find_object_key(file_uuid, "processed") or _find_object_key(file_uuid, "raw")
    if key is None:
        raise FileNotFoundError(f"No image found in MinIO for file_uuid={file_uuid}")

    response = minio_client.get_object(BUCKET_NAME, key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()
