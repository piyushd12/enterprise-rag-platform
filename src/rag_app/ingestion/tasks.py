"""
Celery tasks for asynchronous document ingestion.

Each task is a standalone function (not async) that runs in a Celery worker
process. It imports providers directly since it cannot use FastAPI's
dependency injection. Progress is reported via Celery's update_state()
for live progress tracking in the UI.

Note: Because ingest_file and store operations are async, the sync Celery
task must manage its own asyncio event loop. A single loop is created and
reused for the entire task lifecycle (never closed mid-task).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from celery.utils.time import get_exponential_backoff_interval

from rag_app.config.settings import settings
from rag_app.embeddings.fastembed_provider import FastEmbedProvider
from rag_app.ingestion.pipeline import ingest_file
from rag_app.observability.provider import ObservabilityProvider
from rag_app.queue.celery_app import celery_app
from rag_app.vectorstore.qdrant_store import QdrantStore


def _get_worker_obs() -> ObservabilityProvider:
    """Create an ObservabilityProvider for the worker process.

    Unlike the FastAPI process, the worker constructs its own instances
    (no FastAPI dependency injection).
    """
    return ObservabilityProvider()


def _get_worker_embedder() -> FastEmbedProvider:
    """Create an EmbeddingProvider for the worker process."""
    return FastEmbedProvider()


def _get_worker_store(obs: ObservabilityProvider) -> QdrantStore:
    """Create a VectorStore for the worker process."""
    embedder = _get_worker_embedder()
    return QdrantStore(obs=obs, embedding_dimension=embedder.dimension())


@celery_app.task(
    bind=True,
    name="rag_app.ingestion.tasks.ingest_document",
    max_retries=3,
    default_retry_delay=5,
    retry_backoff=True,
    retry_backoff_max=60,
    acks_late=True,
)
def ingest_document(self, file_path: str, source_id: str, metadata: dict):
    """Async document ingestion task.

    Steps:
        1. Load file → chunk → embed → upsert to Qdrant
        2. Report progress via update_state()
        3. Increment collection version counter on success (invalidates answer cache)
        4. Log lifecycle events to Logfire
        5. Retry with exponential backoff on failure

    Args:
        file_path: Filename of the temp file within settings.ingest_tmp_dir
            (resolved locally, not an absolute path — the API process and
            this worker may run in different containers/filesystems).
        source_id: Unique identifier for this document.
        metadata: Additional metadata (filename, doc_type, etc.).
    """
    obs = _get_worker_obs()
    task_id = self.request.id
    filename = metadata.get("filename", "unknown")
    resolved_path = Path(settings.ingest_tmp_dir).resolve() / file_path

    # --- Task started ---
    obs.log_event(
        "ingestion.task.started",
        {"task_id": task_id, "filename": filename, "source_id": source_id},
    )

    try:
        self.update_state(
            state="STARTED",
            meta={
                "message": f"Starting ingestion of {filename}",
                "chunks_processed": 0,
                "total_chunks": 0,
            },
        )

        # Check file exists
        if not resolved_path.exists():
            raise FileNotFoundError(f"File not found: {resolved_path}")

        # Get provider instances
        embedder = _get_worker_embedder()
        store = _get_worker_store(obs)

        # Run ingestion in a dedicated event loop (reused for the whole task)
        start = time.perf_counter()
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(store.ensure_collection())
            result = loop.run_until_complete(
                ingest_file(
                    file_path=str(resolved_path),
                    embedding_provider=embedder,
                    vector_store=store,
                    obs=obs,
                    filename=filename,
                    source_id=source_id,
                )
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        # --- Task succeeded ---
        obs.log_event(
            "ingestion.task.succeeded",
            {
                "task_id": task_id,
                "filename": filename,
                "chunks_stored": result["chunk_count"],
                "duration_ms": duration_ms,
            },
        )

        # Increment collection version to invalidate answer cache
        from rag_app.caching.redis_cache import RedisCache
        cache = RedisCache(obs=obs)
        cache.increment_collection_version()

        # Clean up temp file
        resolved_path.unlink(missing_ok=True)

        return {
            "source_id": result["source_id"],
            "filename": result["filename"],
            "chunk_count": result["chunk_count"],
            "doc_type": result["doc_type"],
            "duration_ms": duration_ms,
        }

    except Exception as exc:
        # Log failure before retrying
        obs.log_error(
            "ingestion.task.failed",
            {
                "task_id": task_id,
                "filename": filename,
                "error": str(exc),
                "attempt": self.request.retries + 1,
            },
        )

        # Retry with exponential backoff. The task's retry_backoff/
        # retry_backoff_max options only apply to Celery's automatic
        # autoretry_for-driven retries -- a manual self.retry() call like
        # this one ignores them and falls back to a flat default_retry_delay
        # every time unless the countdown is computed explicitly, so it's
        # done here instead.
        # NOTE: temp file is NOT deleted here — a retry may need it again.
        # It is cleaned up only on final task completion or when the
        # task is definitively marked FAILURE after all retries are exhausted.
        countdown = get_exponential_backoff_interval(
            factor=self.default_retry_delay,
            retries=self.request.retries,
            maximum=self.retry_backoff_max,
        )
        try:
            raise self.retry(exc=exc, countdown=countdown)
        except Exception:
            # After the final retry attempt, self.retry() raises MaxRetriesExceededError.
            # Clean up the temp file and re-raise.
            resolved_path.unlink(missing_ok=True)
            raise
