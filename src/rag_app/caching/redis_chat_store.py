"""
RedisChatStore — ChatStore implementation for persisted multi-turn chats.

Separate from RedisCache: this is durable conversation history (no TTL),
not a performance cache. A database-backed ChatStore can replace this
later without touching any call site, since routes depend on the
ChatStore interface, not this class. Redis key design:

- chat:meta:{id}      hash    {title, updated_at}
- chat:messages:{id}  list    JSON-encoded message dicts, oldest first
- chats:index         zset    member=id, score=updated_at -- lets list_chats()
                               return most-recently-active chats first without
                               scanning all chat:meta:* keys
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import redis as redis_lib

from rag_app.config.settings import settings
from rag_app.core.interfaces import ChatStore


class RedisChatStore(ChatStore):
    """Redis-backed ChatStore."""

    INDEX_KEY = "chats:index"

    def __init__(self, redis_url: str | None = None) -> None:
        self._redis = redis_lib.from_url(
            redis_url or settings.redis_url,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )

    async def create_chat(self, title: str) -> str:
        chat_id = str(uuid.uuid4())
        now = time.time()
        self._redis.hset(f"chat:meta:{chat_id}", mapping={"title": title, "updated_at": now})
        self._redis.zadd(self.INDEX_KEY, {chat_id: now})
        return chat_id

    async def list_chats(self) -> list[dict[str, Any]]:
        ids = self._redis.zrevrange(self.INDEX_KEY, 0, -1)
        chats = []
        for chat_id in ids:
            meta = self._redis.hgetall(f"chat:meta:{chat_id}")
            if meta:
                chats.append(
                    {"id": chat_id, "title": meta.get("title", ""), "updated_at": meta.get("updated_at", "")}
                )
        return chats

    async def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        meta = self._redis.hgetall(f"chat:meta:{chat_id}")
        if not meta:
            return None
        raw_messages = self._redis.lrange(f"chat:messages:{chat_id}", 0, -1)
        return {
            "id": chat_id,
            "title": meta.get("title", ""),
            "messages": [json.loads(m) for m in raw_messages],
        }

    async def append_message(self, chat_id: str, message: dict[str, Any]) -> None:
        self._redis.rpush(f"chat:messages:{chat_id}", json.dumps(message))
        now = time.time()
        self._redis.hset(f"chat:meta:{chat_id}", "updated_at", now)
        self._redis.zadd(self.INDEX_KEY, {chat_id: now})
