from app.services.extractors.pdf_extractor import PDFExtractor
from app.services.extractors.docx_extractor import DOCXExtractor
from app.services.extractors.factory import get_extractor, SUPPORTED_TYPES


def test_pdf_extractor_extracts_text(sample_pdf_bytes: bytes):
    extractor = PDFExtractor()
    result = extractor.extract(sample_pdf_bytes)

    assert "Eiffel Tower" in result["text"]
    assert "Paris" in result["text"]
    assert result["page_count"] == 1
    assert len(result["pages"]) == 1
    assert result["pages"][0]["page_num"] == 1


def test_pdf_extractor_labels_pages(sample_pdf_bytes: bytes):
    extractor = PDFExtractor()
    result = extractor.extract(sample_pdf_bytes)
    assert "[Page 1]" in result["text"]


def test_docx_extractor_extracts_text(sample_docx_bytes: bytes):
    extractor = DOCXExtractor()
    result = extractor.extract(sample_docx_bytes)

    assert "Artificial intelligence" in result["text"]
    assert "Machine learning" in result["text"]
    assert result["page_count"] == 1


def test_factory_returns_correct_extractor():
    pdf_extractor = get_extractor("application/pdf")
    assert isinstance(pdf_extractor, PDFExtractor)

    docx_extractor = get_extractor(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert isinstance(docx_extractor, DOCXExtractor)


def test_factory_raises_for_unsupported_type():
    import pytest
    with pytest.raises(ValueError, match="Unsupported file type"):
        get_extractor("application/zip")


def test_factory_normalizes_content_type():
    extractor = get_extractor("application/pdf; charset=utf-8")
    assert isinstance(extractor, PDFExtractor)