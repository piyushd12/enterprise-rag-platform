"""
FastAPI application factory.

- Configures Logfire and LangSmith at startup
- Auto-instruments FastAPI with Logfire
- Registers all route modules
- Ensures Qdrant collection exists on startup
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import logfire
from fastapi import FastAPI

from rag_app.api.dependencies import get_obs, get_vector_store
from rag_app.api.routes import chat, health, ingest
from rag_app.observability import configure_langsmith, configure_logfire


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

    # Register routes
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(ingest.router)

    return app


# Module-level app instance for uvicorn
app = create_app()
