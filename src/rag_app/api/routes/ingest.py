"""
Document ingestion endpoint.

Phase 2: Asynchronous ingestion via Celery — enqueues a task and returns
a task_id immediately. GET /ingest/status/{task_id} polls task state with
progress metadata for live progress tracking.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from rag_app.api.dependencies import get_obs, get_queue
from rag_app.api.rate_limit import limiter
from rag_app.api.schemas import IngestResponse, TaskStatusResponse
from rag_app.config.settings import settings
from rag_app.core.interfaces import TaskQueue
from rag_app.observability.provider import ObservabilityProvider

router = APIRouter(tags=["ingestion"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB


@router.post("/ingest", response_model=IngestResponse)
@limiter.limit("10/minute")
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="Document file (PDF, TXT, or MD)"),
    obs: ObservabilityProvider = Depends(get_obs),
    queue: TaskQueue = Depends(get_queue),
) -> IngestResponse:
    """Upload and ingest a document into the vector store (async via Celery).

    The file is saved to a temporary location, a Celery task is enqueued,
    and the task_id is returned immediately. Use GET /ingest/status/{task_id}
    to poll progress and get results.

    Supports PDF, TXT, and MD files.
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
        # Save uploaded file into the shared ingest tmp dir (not the system /tmp)
        # so a Celery worker in a separate container/filesystem can find it —
        # only the filename crosses the process boundary, never an absolute path.
        tmp_dir = Path(settings.ingest_tmp_dir).resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # Stream to disk in bounded chunks rather than reading the whole
        # upload into memory first -- an unbounded `await file.read()`
        # would let a single oversized upload exhaust worker memory before
        # any size check could reject it.
        hasher = hashlib.sha256()
        size = 0
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=ext, prefix="rag_ingest_", dir=tmp_dir
        ) as tmp:
            tmp_filename = Path(tmp.name).name
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    tmp.close()
                    (tmp_dir / tmp_filename).unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB)",
                    )
                hasher.update(chunk)
                tmp.write(chunk)

        # Generate source_id (stable hash of the file content)
        source_id = hasher.hexdigest()[:16]

        # Prepare metadata
        metadata = {
            "filename": filename,
            "doc_type": ext.lstrip("."),
            "source_id": source_id,
        }

        # Enqueue the Celery task
        try:
            task_id = queue.enqueue(
                "rag_app.ingestion.tasks.ingest_document",
                file_path=tmp_filename,
                source_id=source_id,
                metadata=metadata,
            )
        except Exception as e:
            # Cleanup on enqueue failure
            (tmp_dir / tmp_filename).unlink(missing_ok=True)
            obs.log_error("ingest.enqueue.failed", {"filename": filename, "error": str(e)})
            raise HTTPException(status_code=503, detail=f"Task queue unavailable: {e}")

    return IngestResponse(
        task_id=task_id,
        status="queued",
        filename=filename,
        message=f"Ingestion task enqueued for {filename}",
    )


@router.get("/ingest/status/{task_id}", response_model=TaskStatusResponse)
async def get_ingest_status(
    task_id: str,
    obs: ObservabilityProvider = Depends(get_obs),
    queue: TaskQueue = Depends(get_queue),
) -> TaskStatusResponse:
    """Poll the status of an ingestion task.

    Returns the current state (queued/started/retried/succeeded/failed),
    any error message, and progress metadata with chunk counts.
    """
    try:
        status = queue.get_status(task_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query task status: {e}")

    return TaskStatusResponse(
        task_id=status["task_id"],
        status=status["status"],
        result=status.get("result"),
        error=status.get("error"),
        progress=status.get("progress"),
    )
