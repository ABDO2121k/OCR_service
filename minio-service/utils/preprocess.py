import cv2
import numpy as np

TARGET_WIDTH = 2000


def preprocess_image(raw_bytes: bytes) -> bytes:
    """Prepares a mobile photo of a Carte Grise for OCR: grayscale,
    deskew, contrast normalization, and resize to an OCR-friendly width."""
    image = _bytes_to_cv2(raw_bytes)
    image = _to_grayscale(image)
    image = _deskew(image)
    image = _normalize_contrast(image)
    image = _resize(image, TARGET_WIDTH)
    return _cv2_to_bytes(image)


def _bytes_to_cv2(raw_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(raw_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image bytes")
    return image


def _cv2_to_bytes(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Could not encode processed image")
    return buffer.tobytes()


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _deskew(image: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(image < 200))
    if len(coords) == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _normalize_contrast(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def _resize(image: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= target_width:
        return image
    scale = target_width / w
    return cv2.resize(image, (target_width, int(h * scale)), interpolation=cv2.INTER_AREA)
