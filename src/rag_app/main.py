"""
FastAPI application factory.

- Configures Logfire and LangSmith at startup
- Auto-instruments FastAPI with Logfire
- Registers all route modules
- Serves the static chat UI (src/rag_app/static/) at "/"
- Ensures Qdrant collection exists on startup
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import logfire
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from rag_app.api.dependencies import get_keyword_search, get_obs, get_vector_store
from rag_app.api.rate_limit import limiter
from rag_app.api.routes import chat, chats, documents, evaluate, health, ingest
from rag_app.observability import configure_langsmith, configure_logfire

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown hooks."""
    obs = get_obs()
    obs.log_event("app.startup", {"version": "0.1.0"})

    # Ensure vector store collection exists
    try:
        store = get_vector_store()
        await store.ensure_collection()
        obs.log_event("app.qdrant.ready", {"collection": "documents"})
    except Exception as e:
        obs.log_warning("app.qdrant.unavailable", {"error": str(e)})

    # Prebuild the BM25 index (now used by default in /chat, alongside the
    # reranker) so the first real chat request isn't the one that pays for
    # scrolling the whole collection and tokenizing it.
    try:
        get_keyword_search().rebuild()
        obs.log_event("app.bm25.ready")
    except Exception as e:
        obs.log_warning("app.bm25.unavailable", {"error": str(e)})

    yield

    # --- Shutdown ---
    obs.log_event("app.shutdown")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Bootstrap observability BEFORE anything else
    configure_logfire()
    configure_langsmith()

    app = FastAPI(
        title="RAG Application",
        description="Enterprise-grade Retrieval-Augmented Generation API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Auto-instrument FastAPI with Logfire (request/response, latency, status)
    logfire.instrument_fastapi(app)

    # Per-IP rate limiting (see api/rate_limit.py) on /chat and /ingest
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # Register routes
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(ingest.router)
    app.include_router(evaluate.router)
    app.include_router(documents.router)
    app.include_router(chats.router)

    # Serve the chat UI. Mounted last (and at "/") so it only catches
    # requests the API routes above didn't already claim.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


# Module-level app instance for uvicorn
app = create_app()
