"""
Logfire configuration and helpers.

Logfire handles ALL non-LLM-pipeline observability:
- FastAPI request/response (auto-instrumentation)
- Celery task lifecycle
- Redis cache hit/miss events
- LLM fallback events
- Startup/config validation
- Unhandled exceptions
"""

import logfire

from rag_app.config.settings import settings


def configure_logfire() -> None:
    """Initialize Logfire with the application token.

    Must be called once at app startup, before any instrumented code runs.
    """
    logfire.configure(
        token=settings.logfire_token if settings.logfire_token else None,
        service_name="rag-app",
        send_to_logfire="if-token-present",
    )
    logfire.info(
        "Logfire configured",
        service="rag-app",
        qdrant_url=settings.qdrant_url,
        redis_url=settings.redis_url,
        embedding_model=settings.embedding_model,
    )
