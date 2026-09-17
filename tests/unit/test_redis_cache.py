"""Unit tests for RedisCache hit/miss/invalidation behavior (mocked redis client).

No real Redis connection is made -- redis.from_url is patched to return a
MagicMock so these tests only exercise RedisCache's own key-building,
serialization, and cache-aside logic.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from rag_app.caching.redis_cache import RedisCache


def _make_cache(fake_redis: MagicMock) -> RedisCache:
    with patch("rag_app.caching.redis_cache.redis_lib.from_url", return_value=fake_redis):
        return RedisCache(obs=None, redis_url="redis://fake:6379/0")


async def test_get_answer_cache_miss_returns_none():
    fake_redis = MagicMock()
    fake_redis.get.return_value = None
    fake_redis.setnx.return_value = True
    cache = _make_cache(fake_redis)

    result = await cache.get_answer("What is RAG?")

    assert result is None


async def test_get_answer_cache_hit_returns_deserialized_value():
    fake_redis = MagicMock()
    stored = {"answer": "RAG stands for Retrieval-Augmented Generation.", "sources": []}

    def fake_get(key):
        if key == "rag:collection_version":
            return "1"
        return json.dumps(stored)

    fake_redis.get.side_effect = fake_get
    cache = _make_cache(fake_redis)

    result = await cache.get_answer("What is RAG?")

    assert result == stored


async def test_set_answer_stores_serialized_value_with_ttl():
    fake_redis = MagicMock()
    fake_redis.get.return_value = "1"
    cache = _make_cache(fake_redis)

    await cache.set_answer("What is RAG?", "It's retrieval-augmented generation.", [])

    assert fake_redis.setex.call_count == 1
    key, ttl, payload = fake_redis.setex.call_args[0]
    assert key.startswith("ans:")
    assert ttl == cache._answer_ttl
    assert json.loads(payload)["answer"] == "It's retrieval-augmented generation."


async def test_incrementing_collection_version_changes_the_answer_cache_key():
    """A collection-version bump (fired on new ingestion) must change the
    answer cache key for the same query, so old cached answers are
    effectively invalidated without scanning/deleting any keys."""
    fake_redis = MagicMock()
    fake_redis.get.return_value = "1"
    cache = _make_cache(fake_redis)

    key_v1, _ = cache._build_answer_key("same query")

    fake_redis.get.return_value = "2"
    key_v2, _ = cache._build_answer_key("same query")

    assert key_v1 != key_v2


async def test_cache_read_failure_is_swallowed_not_raised():
    """A broken Redis connection must degrade to a cache miss, never break
    the request -- get() catches internally and returns None."""
    fake_redis = MagicMock()
    fake_redis.get.side_effect = ConnectionError("redis unreachable")
    cache = _make_cache(fake_redis)

    result = await cache.get("some:key")

    assert result is None
