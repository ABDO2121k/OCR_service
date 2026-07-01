import numpy as np
import cv2
from paddleocr import PaddleOCR

_ocr_engine = None


def _get_engine() -> PaddleOCR:
    global _ocr_engine
    if _ocr_engine is None:
        # 'latin' covers the French/Dutch/German Latin-script text on the Carte Grise
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="latin", show_log=False)
    return _ocr_engine


def extract_text(image_bytes: bytes) -> str:
    """Runs PaddleOCR on the image and returns the raw extracted text,
    one line per detected text block, ordered top-to-bottom."""
    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image for OCR")

    engine = _get_engine()
    result = engine.ocr(image, cls=True)

    lines = []
    for page in result or []:
        for detection in page or []:
            text, _confidence = detection[1]
            lines.append(text)
    return "\n".join(lines)
