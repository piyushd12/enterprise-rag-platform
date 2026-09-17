"""
Basic per-IP rate limiting for public endpoints.

Uses slowapi (backed by the `limits` library) with Redis as the shared
counter store, so limits hold even if the API runs as multiple
processes/replicas. Applied to /chat and /ingest -- the two endpoints
that trigger a real LLM call or heavy async processing, and the ones
most exposed to accidental hammering or abuse.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from rag_app.config.settings import settings

limiter = Limiter(key_func=get_remote_address, storage_uri=settings.redis_url)
