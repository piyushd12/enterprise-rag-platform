import logging
import uuid
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

from app.core.config import get_settings
from app.services.embedder import EMBEDDING_DIM

settings = get_settings()
logger = logging.getLogger(__name__)

COLLECTION_NAME = "document_chunks"


class VectorStoreService:

    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url)
        logger.info(f"Connected to Qdrant at {settings.qdrant_url}")

    def ensure_collection_exists(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]

        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: '{COLLECTION_NAME}'")

            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="workspace_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Created payload index on 'workspace_id'")
        else:
            logger.info(f"Qdrant collection '{COLLECTION_NAME}' already exists")

    def upsert_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> list[str]:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        points = []
        point_ids = []
        for chunk, vector in zip(chunks, embeddings):
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "chunk_id": chunk["id"],
                        "document_id": chunk["document_id"],
                        "workspace_id": chunk["workspace_id"],
                        "chunk_text": chunk["chunk_text"],
                        "chunk_index": chunk["chunk_index"],
                        "page_num": chunk["page_num"],
                        "token_count": chunk["token_count"],
                    },
                )
            )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )
        logger.info(f"Upserted {len(points)} vectors into Qdrant")

        # Return the point IDs (was unreachable before due to early return)
        return point_ids

    def search(
        self,
        query_vector: list[float],
        workspace_id: str,
        top_k: int = 10,
        document_id: str | None = None,
    ) -> list[dict]:

        filter_conditions = [
            FieldCondition(
                key="workspace_id",
                match=MatchValue(value=workspace_id),
            )
        ]

        if document_id:
            filter_conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=document_id),
                )
            )

        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=Filter(must=filter_conditions),
            limit=top_k,
            with_payload=True,
            # Removed score_threshold=0.3 — it silently drops borderline-relevant chunks.
            # RRF fusion handles ranking quality without needing a hard cutoff.
        )

        return [
            {
                "chunk_text": r.payload["chunk_text"],
                "chunk_id": r.payload["chunk_id"],
                "document_id": r.payload["document_id"],
                "page_num": r.payload["page_num"],
                "chunk_index": r.payload["chunk_index"],
                "score": round(r.score, 4),
            }
            for r in results
        ]

    def delete_document_vectors(self, document_id: str, workspace_id: str) -> None:
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id)),
                    FieldCondition(key="workspace_id", match=MatchValue(value=workspace_id)),
                ]
            ),
        )
        logger.info(f"Deleted Qdrant vectors for document_id={document_id}")


vector_store = VectorStoreService()