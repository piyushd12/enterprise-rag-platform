"""
Document ingestion endpoint.

Phase 1: Synchronous ingestion (blocks until complete).
Phase 2: Will be replaced with async Celery task + status polling.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from rag_app.api.dependencies import (
    get_embedding_provider,
    get_obs,
    get_vector_store,
)
from rag_app.api.schemas import IngestResponse
from rag_app.core.interfaces import EmbeddingProvider, VectorStore
from rag_app.ingestion.pipeline import ingest_file
from rag_app.observability.provider import ObservabilityProvider

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="Document file (PDF, TXT, or MD)"),
    obs: ObservabilityProvider = Depends(get_obs),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    vector_store: VectorStore = Depends(get_vector_store),
) -> IngestResponse:
    """Upload and ingest a document into the vector store.

    Accepts PDF, TXT, and MD files. The file is chunked, embedded,
    and upserted into Qdrant with metadata.
    """
    # Validate file type
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: .pdf, .txt, .md",
        )

    with obs.span("ingest.endpoint", {"filename": filename}):
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=ext, prefix="rag_ingest_"
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Ensure collection exists
            await vector_store.ensure_collection()

            # Run ingestion pipeline
            result = await ingest_file(
                file_path=tmp_path,
                embedding_provider=embedding_provider,
                vector_store=vector_store,
                obs=obs,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            obs.log_error("ingest.failed", {"filename": filename, "error": str(e)})
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

    return IngestResponse(
        source_id=result["source_id"],
        filename=result["filename"],
        chunk_count=result["chunk_count"],
        doc_type=result["doc_type"],
    )
