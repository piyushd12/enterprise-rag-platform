"""
Document listing endpoint — powers the "uploaded documents" view in the UI.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rag_app.api.dependencies import get_vector_store
from rag_app.api.schemas import DocumentInfo, DocumentsResponse
from rag_app.core.interfaces import VectorStore

router = APIRouter(tags=["documents"])


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(
    vector_store: VectorStore = Depends(get_vector_store),
) -> DocumentsResponse:
    """List every distinct document currently ingested, newest first."""
    sources = await vector_store.list_sources()
    return DocumentsResponse(documents=[DocumentInfo(**s) for s in sources])
