import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.dependencies import get_current_user, get_current_workspace
from app.models.user import User
from app.models.workspace import Workspace
from app.schemas.search import SearchRequest, SearchResponse, ChunkResult
from app.services.embedder import embedder
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspaces/{workspace_id}",
    tags=["Search"],
)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic search across workspace documents",
)
async def search_documents(
    workspace_id: str,
    request: SearchRequest,
    current_user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    db: AsyncSession = Depends(get_db),
):
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Search query cannot be empty",
        )

    logger.info(
        f"Search: workspace={workspace_id}, "
        f"query='{request.query[:50]}', top_k={request.top_k}"
    )

    query_vector = embedder.embed_query(request.query)

    raw_results = vector_store.search(
        query_vector=query_vector,
        workspace_id=workspace_id,
        top_k=request.top_k,
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