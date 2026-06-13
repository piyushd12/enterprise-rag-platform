import logging
import fitz   # PyMuPDF 

logger = logging.getLogger(__name__)


class PDFExtractor:
    """
    Extracts text from PDF files using PyMuPDF.
    Handles text-based PDFs well. For scanned-image PDFs (no text layer),
    the text will be empty, the image extractor handles those.
    """

    def extract(self, file_bytes: bytes) -> dict:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages = []
        page_boundaries = []
        running_char_count = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            page_label = f"[Page {page_num + 1}]\n"
            page_text = page_label + text

            page_boundaries.append(running_char_count)
            running_char_count += len(page_text) + 2

            pages.append({
                "page_num": page_num + 1,
                "text": text,
            })

        full_text = "\n\n".join(
            f"[Page {p['page_num']}]\n{p['text']}"
            for p in pages
            if p["text"]
        )

        logger.info(f"PDF extracted: {len(doc)} pages, {len(full_text)} characters")

        return {
            "text": full_text,
            "page_count": len(doc),
            "pages": pages,
            "page_boundaries": page_boundaries,
        }