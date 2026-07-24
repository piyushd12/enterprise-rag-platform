import io
import logging
import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)


class ImageExtractor:
    def extract(self, file_bytes: bytes) -> dict:
        image = Image.open(io.BytesIO(file_bytes))

        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        text = pytesseract.image_to_string(image, lang="eng").strip()
        logger.info(f"Image OCR extracted: {len(text)} characters")

        return {
            "text": text,
            "page_count": 1,
            "pages": [{"page_num": 1, "text": text}],
        }