"""Unit tests for RedisChatStore delete behavior (mocked redis client)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag_app.caching.redis_chat_store import RedisChatStore


def _make_store(fake_redis: MagicMock) -> RedisChatStore:
    with patch("rag_app.caching.redis_chat_store.redis_lib.from_url", return_value=fake_redis):
        return RedisChatStore(redis_url="redis://fake:6379/0")


async def test_delete_chat_removes_keys_and_index_entry():
    fake_redis = MagicMock()
    fake_redis.delete.return_value = 2
    store = _make_store(fake_redis)

    assert await store.delete_chat("abc") is True
    fake_redis.delete.assert_called_once_with("chat:meta:abc", "chat:messages:abc")
    fake_redis.zrem.assert_called_once_with("chats:index", "abc")


async def test_delete_missing_chat_returns_false():
    fake_redis = MagicMock()
    fake_redis.delete.return_value = 0
    store = _make_store(fake_redis)

    assert await store.delete_chat("nope") is False


async def test_append_to_deleted_chat_does_not_resurrect_it():
    fake_redis = MagicMock()
    fake_redis.exists.return_value = 0
    store = _make_store(fake_redis)

    await store.append_message("gone", {"role": "assistant", "content": "late"})

    fake_redis.rpush.assert_not_called()
    fake_redis.hset.assert_not_called()
    fake_redis.zadd.assert_not_called()
