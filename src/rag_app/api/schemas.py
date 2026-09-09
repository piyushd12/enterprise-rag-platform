"""
Pydantic request/response schemas for the FastAPI endpoints.

These schemas define the API contract — they are independent of any
concrete implementation and can be used by the Streamlit client too.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    query: str = Field(..., min_length=1, max_length=2000, description="The user's question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    filters: dict | None = Field(default=None, description="Optional payload filters, e.g. {'doc_type': 'pdf'}")


class SourceChunk(BaseModel):
    """A single source chunk returned alongside the answer."""

    content: str
    source_id: str
    chunk_index: int
    score: float
    filename: str = ""


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    request_id: str
    answer: str
    sources: list[SourceChunk]
    cached: bool = False
    llm_provider: str
    llm_model: str
    latency_ms: float


# ---------------------------------------------------------------------------
# /ingest
# ---------------------------------------------------------------------------


class IngestResponse(BaseModel):
    """Response body for POST /ingest (async, returns task id)."""

    task_id: str
    status: str = "queued"
    filename: str
    message: str = "Ingestion task enqueued"


class TaskStatusResponse(BaseModel):
    """Response body for GET /ingest/status/{task_id}."""

    task_id: str
    status: Literal["queued", "started", "retried", "succeeded", "failed"]
    result: dict | None = None
    error: str | None = None
    progress: dict | None = None  # e.g. {"chunks_processed": 5, "total_chunks": 10}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str  # "healthy" | "degraded"
    qdrant: bool
    redis: bool
    version: str
