"""
Qdrant-backed vector store implementation.

Supports both local Docker Qdrant and Qdrant Cloud (set QDRANT_API_KEY).
Collection is auto-created on first use with COSINE distance and keyword
indexes on source_id and doc_type for filtered retrieval.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from rag_app.config.settings import settings
from rag_app.core.interfaces import RetrievedChunk, VectorStore
from rag_app.observability.provider import ObservabilityProvider


class QdrantStore(VectorStore):
    """VectorStore implementation backed by Qdrant."""

    def __init__(
        self,
        obs: ObservabilityProvider,
        embedding_dimension: int = 384,
        collection_name: str | None = None,
    ) -> None:
        self._obs = obs
        self._collection_name = collection_name or settings.qdrant_collection_name
        self._dimension = embedding_dimension

        # Connect — use API key for cloud, plain URL for local Docker
        if settings.qdrant_api_key:
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )
        else:
            self._client = QdrantClient(url=settings.qdrant_url)

    async def ensure_collection(self) -> None:
        """Create collection with COSINE distance and payload indexes if absent."""
        collections = self._client.get_collections().collections
        existing_names = {c.name for c in collections}

        if self._collection_name in existing_names:
            self._obs.log_event(
                "qdrant.collection.exists",
                {"collection": self._collection_name},
            )
            return

        self._client.create_collection(
            collection_name=self._collection_name,
            vectors_config=VectorParams(
                size=self._dimension,
                distance=Distance.COSINE,
            ),
        )

        # Keyword indexes for filtered search
        for field in ("source_id", "doc_type"):
            self._client.create_payload_index(
                collection_name=self._collection_name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )

        self._obs.log_event(
            "qdrant.collection.created",
            {
                "collection": self._collection_name,
                "dimension": self._dimension,
                "distance": "cosine",
            },
        )

    async def add_documents(self, chunks: list[dict[str, Any]]) -> None:
        """Upsert document chunks into Qdrant.

        Each chunk dict must have: id, embedding, content, metadata.
        """
        with self._obs.span(
            "qdrant.upsert", {"chunk_count": len(chunks)}
        ) as span_data:
            points = [
                PointStruct(
                    id=chunk.get("id", str(uuid.uuid4())),
                    vector=chunk["embedding"],
                    payload={
                        "content": chunk["content"],
                        "source_id": chunk["metadata"].get("source_id", ""),
                        "chunk_index": chunk["metadata"].get("chunk_index", 0),
                        "doc_type": chunk["metadata"].get("doc_type", ""),
                        **{
                            k: v
                            for k, v in chunk["metadata"].items()
                            if k not in ("source_id", "chunk_index", "doc_type")
                        },
                    },
                )
                for chunk in chunks
            ]

            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            span_data["points_upserted"] = len(points)

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Search for the top_k most similar chunks."""
        qdrant_filter = None
        if filters:
            must_conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=must_conditions)

        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
        )

        return [
            RetrievedChunk(
                content=hit.payload.get("content", ""),
                score=hit.score,
                metadata={
                    k: v for k, v in hit.payload.items() if k != "content"
                },
            )
            for hit in results.points
        ]

    async def delete_by_source(self, source_id: str) -> None:
        """Delete all chunks belonging to a source document."""
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source_id",
                        match=MatchValue(value=source_id),
                    )
                ]
            ),
        )
        self._obs.log_event(
            "qdrant.delete_by_source",
            {"source_id": source_id, "collection": self._collection_name},
        )
