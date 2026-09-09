"""
RedisCache — CacheProvider implementation using Redis with cache-aside pattern.

Cache key design:
- Answer cache:  ans:{sha256(normalized_query + collection_version)}  →  JSON(answer + sources)
- Embedding cache: emb:{sha256(text)}  →  JSON(embedding vector)

Query normalization: lowercase, strip whitespace, collapse runs of whitespace.
Collection version: Redis INCR counter — incremented on every ingestion so all
answer cache entries are automatically invalidated without scanning keys.

TTL is set per cache type via settings (answers: 1h, embeddings: 24h).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import redis as redis_lib

from rag_app.config.settings import settings
from rag_app.observability.provider import ObservabilityProvider

# Redis key for the collection version counter
COLLECTION_VERSION_KEY = "rag:collection_version"


def normalize_query(text: str) -> str:
    """Normalize a query for cache key generation.

    Steps: lowercase, strip leading/trailing whitespace,
    collapse internal runs of whitespace to a single space.
    """
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)


def _make_cache_key(prefix: str, raw: str) -> str:
    """Build a SHA-256 prefixed cache key."""
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}:{digest}"


class RedisCache:
    """Redis-backed CacheProvider with cache-aside pattern and observability.

    All cache operations are logged to Logfire via ObservabilityProvider.
    """

    def __init__(
        self,
        obs: ObservabilityProvider | None = None,
        redis_url: str | None = None,
    ):
        self._redis = redis_lib.from_url(
            redis_url or settings.redis_url,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        self._obs = obs
        self._answer_ttl = settings.cache_ttl_answers
        self._embedding_ttl = settings.cache_ttl_embeddings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_collection_version(self) -> int:
        """Get the current collection version counter. Creates with value 1 if missing."""
        val = self._redis.get(COLLECTION_VERSION_KEY)
        if val is None:
            # Use SETNX to avoid race condition on first call
            if self._redis.setnx(COLLECTION_VERSION_KEY, 1):
                return 1
            return int(self._redis.get(COLLECTION_VERSION_KEY) or 1)
        return int(val)

    def increment_collection_version(self) -> int:
        """Increment the collection version counter after ingestion.

        This automatically invalidates all answer cache entries because
        the version component of the cache key changes.
        """
        new_version = self._redis.incr(COLLECTION_VERSION_KEY)
        if self._obs:
            self._obs.log_event(
                "cache.version.incremented",
                {"new_version": new_version},
            )
        return new_version

    def _build_answer_key(self, query: str) -> str:
        """Build answer cache key: normalizes query + appends collection version."""
        normalized = normalize_query(query)
        version = self._get_collection_version()
        raw = f"{normalized}:{version}"
        return _make_cache_key("ans", raw), normalized

    def _build_embedding_key(self, text: str) -> str:
        """Build embedding cache key from text content."""
        return _make_cache_key("emb", text)

    # ------------------------------------------------------------------
    # CacheProvider interface — get / set / invalidate_pattern
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any | None:
        """Retrieve a cached value by key."""
        try:
            raw = self._redis.get(key)
            if raw is not None:
                if self._obs:
                    self._obs.log_event(
                        "cache.lookup",
                        {"cache_key_prefix": key.split(":")[0], "hit": True},
                    )
                return json.loads(raw)
            else:
                if self._obs:
                    self._obs.log_event(
                        "cache.lookup",
                        {"cache_key_prefix": key.split(":")[0], "hit": False},
                    )
                return None
        except Exception:
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value in the cache with optional TTL in seconds."""
        try:
            serialized = json.dumps(value)
            ttl_to_use = ttl or self._answer_ttl
            self._redis.setex(key, ttl_to_use, serialized)
            if self._obs:
                self._obs.log_event(
                    "cache.store",
                    {
                        "cache_key_prefix": key.split(":")[0],
                        "ttl_seconds": ttl_to_use,
                    },
                )
        except Exception:
            pass  # Cache failures should never break the pipeline

    async def invalidate_pattern(self, pattern: str) -> int:
        """Delete all keys matching a pattern. Returns count of deleted keys."""
        try:
            keys = list(self._redis.scan_iter(match=pattern, count=1000))
            if keys:
                deleted = self._redis.delete(*keys)
                if self._obs:
                    self._obs.log_event(
                        "cache.invalidate",
                        {"pattern": pattern, "deleted_count": deleted},
                    )
                return deleted
            return 0
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Convenience methods for the RAG pipeline
    # ------------------------------------------------------------------

    async def get_answer(self, query: str) -> dict | None:
        """Look up a cached answer for the given query.

        Returns deserialized dict with 'answer' and 'sources' keys, or None.
        """
        key, _normalized = self._build_answer_key(query)
        return await self.get(key)

    async def set_answer(
        self, query: str, answer: str, sources: list[dict]
    ) -> None:
        """Store a generated answer in the cache."""
        key, _normalized = self._build_answer_key(query)
        value = {"answer": answer, "sources": sources}
        await self.set(key, value, ttl=self._answer_ttl)

    async def get_embedding(self, text: str) -> list[float] | None:
        """Look up a cached embedding for the given text."""
        key = self._build_embedding_key(text)
        result = await self.get(key)
        return result  # already a list[float] after json.loads

    async def set_embedding(self, text: str, embedding: list[float]) -> None:
        """Cache an embedding vector."""
        key = self._build_embedding_key(text)
        await self.set(key, embedding, ttl=self._embedding_ttl)

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return self._redis.ping()
        except Exception:
            return False
