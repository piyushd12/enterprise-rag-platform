from app.services.extractors.pdf_extractor import PDFExtractor
from app.services.extractors.docx_extractor import DOCXExtractor
from app.services.extractors.image_extractor import ImageExtractor

SUPPORTED_TYPES: dict[str, type] = {
    "application/pdf": PDFExtractor,

    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DOCXExtractor,
    "application/msword": DOCXExtractor,

    "image/jpeg": ImageExtractor,
    "image/jpg": ImageExtractor,
    "image/png": ImageExtractor,
    "image/tiff": ImageExtractor,
    "image/webp": ImageExtractor,
}


def get_extractor(content_type: str):
    clean_type = content_type.split(";")[0].strip().lower()

    extractor_class = SUPPORTED_TYPES.get(clean_type)
    if extractor_class is None:
        raise ValueError(
            f"Unsupported file type: '{content_type}'. "
            f"Supported: PDF, DOCX, JPEG, PNG, TIFF, WEBP"
        )
    return extractor_class()