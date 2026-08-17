"""
Document loaders for PDF, TXT, and Markdown files.

Returns a list of dicts with 'content' and 'metadata' per document.
"""

from __future__ import annotations

import os
from pathlib import Path

from pypdf import PdfReader


def load_document(file_path: str | Path) -> list[dict]:
    """Load a single document and return its content + metadata.

    Supported formats: .pdf, .txt, .md

    Returns:
        List with a single dict: {'content': str, 'metadata': dict}
        (PDF may return multiple dicts if pages are split.)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()
    filename = path.name
    source_id = _generate_source_id(path)

    if ext == ".pdf":
        return _load_pdf(path, source_id, filename)
    elif ext in (".txt", ".md"):
        return _load_text(path, source_id, filename, ext.lstrip("."))
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def _load_pdf(path: Path, source_id: str, filename: str) -> list[dict]:
    """Load a PDF, concatenating all page text into a single document."""
    reader = PdfReader(str(path))
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    full_text = "\n\n".join(pages_text)
    if not full_text.strip():
        raise ValueError(f"PDF contains no extractable text: {path}")

    return [
        {
            "content": full_text,
            "metadata": {
                "source_id": source_id,
                "filename": filename,
                "doc_type": "pdf",
                "page_count": len(reader.pages),
            },
        }
    ]


def _load_text(
    path: Path, source_id: str, filename: str, doc_type: str
) -> list[dict]:
    """Load a plain text or markdown file."""
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"File is empty: {path}")

    return [
        {
            "content": content,
            "metadata": {
                "source_id": source_id,
                "filename": filename,
                "doc_type": doc_type,
            },
        }
    ]


def _generate_source_id(path: Path) -> str:
    """Generate a stable source_id from the absolute file path."""
    import hashlib

    abs_path = str(path.resolve())
    return hashlib.sha256(abs_path.encode()).hexdigest()[:16]
