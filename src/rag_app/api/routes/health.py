"""
Health check endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rag_app.api.dependencies import get_obs
from rag_app.api.schemas import HealthResponse
from rag_app.config.settings import settings
from rag_app.observability.provider import ObservabilityProvider

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    obs: ObservabilityProvider = Depends(get_obs),
) -> HealthResponse:
    """Check the health of all infrastructure dependencies."""
    qdrant_ok = False
    redis_ok = False

    # Check Qdrant
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, timeout=3)
        client.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    # Check Redis
    try:
        import redis

        r = redis.from_url(settings.redis_url, socket_timeout=3)
        r.ping()
        redis_ok = True
    except Exception:
        pass

    status = "healthy" if (qdrant_ok and redis_ok) else "degraded"

    obs.log_event(
        "health.check",
        {"status": status, "qdrant": qdrant_ok, "redis": redis_ok},
    )

    return HealthResponse(
        status=status,
        qdrant=qdrant_ok,
        redis=redis_ok,
        version="0.1.0",
    )
