# app/routers/search.py
"""
Search endpoints for the RAG platform.

/search           — hybrid search (dense + BM25 + RRF) across workspace documents
/search/stats     — Qdrant collection stats for this workspace
/search/rebuild-bm25 — dev tool: rebuild the in-memory BM25 index from DB
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies import get_current_user, get_current_workspace
from app.models.chunk import DocumentChunk
from app.models.user import User
from app.models.workspace import Workspace
from app.schemas.search import SearchRequest, SearchResponse, ChunkResult

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}",
    tags=["Search"],
)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Hybrid search (dense + BM25 + RRF) across workspace documents",
)
async def search_documents(
    workspace_id: str,
    request: SearchRequest,
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    db: AsyncSession = Depends(get_db),
):
    """
    Semantic + keyword hybrid search over all documents in this workspace.

    Uses dense vector search (Qdrant) combined with BM25 keyword search,
    fused via Reciprocal Rank Fusion. Consistently outperforms dense-only
    search for exact terms, proper nouns, and codes.
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty",
        )

    logger.info(
        f"Search: workspace={workspace_id[:8]}, "
        f"query='{request.query[:50]}', top_k={request.top_k}"
    )

    from app.services.retrieval.hybrid_retriever import hybrid_search

    raw_results = await hybrid_search(
        query=request.query,
        workspace_id=workspace_id,
        top_k=request.top_k,
        dense_top_k=request.top_k * 2,
        bm25_top_k=request.top_k * 2,
        document_id=request.document_id,
    )

    if not raw_results:
        logger.info(f"No results found for query: '{request.query}'")

    chunk_results = [ChunkResult(**r) for r in raw_results]

    return SearchResponse(
        query=request.query,
        results=chunk_results,
        total_found=len(chunk_results),
    )


@router.get(
    "/search/stats",
    summary="Get vector store stats for this workspace",
)
async def get_search_stats(
    workspace_id: str,
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
):
    """Returns the number of indexed chunks, BM25 index status, and collection health."""
    from app.services.vector_store import vector_store, COLLECTION_NAME
    from app.services.retrieval.bm25_index import bm25_index
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    count = vector_store.client.count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="workspace_id",
                    match=MatchValue(value=workspace_id),
                )
            ]
        ),
        exact=False,
    )

    return {
        "workspace_id": workspace_id,
        "indexed_chunks_qdrant": count.count,
        "bm25_index_ready": bm25_index.is_ready,
        "bm25_total_chunks": bm25_index.total_chunks,
        "collection": COLLECTION_NAME,
    }


@router.post(
    "/search/rebuild-bm25",
    summary="Rebuild BM25 index from database (dev tool)",
)
async def rebuild_bm25_index(
    workspace_id: str,
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    db: AsyncSession = Depends(get_db),
):
    """
    Rebuilds the in-memory BM25 index from ALL chunks in the database.

    Call this after uploading new documents during development —
    the BM25 index lives in the FastAPI process and is not automatically
    updated when the Celery worker finishes processing.

    In production this is handled by a Redis pub/sub signal (Week 11).
    """
    from app.services.retrieval.bm25_index import bm25_index

    result = await db.execute(select(DocumentChunk))
    all_chunks = result.scalars().all()

    chunk_dicts = [
        {
            "id": c.id,
            "workspace_id": c.workspace_id,
            "document_id": c.document_id,
            "chunk_text": c.chunk_text,
            "page_num": c.page_num,
            "chunk_index": c.chunk_index,
        }
        for c in all_chunks
    ]

    bm25_index.build(chunk_dicts)

    logger.info(
        f"BM25 index rebuilt manually: {bm25_index.total_chunks} total chunks"
    )

    return {
        "status": "rebuilt",
        "total_chunks_indexed": bm25_index.total_chunks,
        "note": "BM25 index is global across all workspaces",
    }