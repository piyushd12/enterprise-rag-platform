# app/services/retrieval/hybrid_retriever.py
"""
Hybrid retrieval: dense vector search (Qdrant) + sparse BM25 search,
fused with Reciprocal Rank Fusion (RRF).

Why hybrid beats either alone:
- Dense search: great for semantic similarity, weak on exact rare tokens
- BM25 search: great for exact keywords/proper nouns, weak on paraphrases
- RRF: rewards chunks that rank well in BOTH — the consensus candidates

RRF formula (Cormack et al., 2009):
    score(chunk) = Σ  1 / (k + rank_in_list)
    where k = 60 (standard constant)

Chunks appearing in both lists accumulate score from each — no score
normalisation needed since only rank positions are used.
"""
import logging

from app.services.embedder import embedder
from app.services.vector_store import vector_store
from app.services.retrieval.bm25_index import bm25_index

logger = logging.getLogger(__name__)

# k=60 from the original RRF paper — keeps scores stable for low-ranked results
RRF_K = 60


def _reciprocal_rank_fusion(
    *ranked_lists: list[dict],
    top_k: int,
) -> list[dict]:
    """
    Merge multiple ranked result lists into one using Reciprocal Rank Fusion.

    Args:
        *ranked_lists: Any number of ranked chunk lists (best first).
                       Each chunk must have a 'chunk_id' field.
        top_k: Maximum number of results to return.

    Returns:
        Merged list ordered by RRF score descending.
        Each chunk's 'score' field is replaced with its RRF score.
        A 'score_type' field is set to 'rrf' for transparency.
    """
    rrf_scores: dict[str, float] = {}   # chunk_id → cumulative RRF score
    chunk_data: dict[str, dict] = {}    # chunk_id → chunk dict for final output

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            chunk_id = chunk["chunk_id"]
            # Add this list's RRF contribution (1-indexed rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            # Store chunk data — prefer the first occurrence (dense scores are normalised)
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = chunk

    # Sort by cumulative RRF score descending
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)

    results = []
    for chunk_id in sorted_ids[:top_k]:
        chunk = dict(chunk_data[chunk_id])   # copy to avoid mutating cached data
        chunk["score"] = round(rrf_scores[chunk_id], 6)
        chunk["score_type"] = "rrf"
        results.append(chunk)

    return results


async def hybrid_search(
    query: str,
    workspace_id: str,
    top_k: int = 10,
    dense_top_k: int = 20,
    bm25_top_k: int = 20,
    document_id: str | None = None,
) -> list[dict]:
    """
    Hybrid retrieval: dense vector search + BM25 → RRF fusion.

    Retrieves more candidates (dense_top_k, bm25_top_k) from each method
    before fusion, ensuring good coverage for the final top_k selection.

    Args:
        query: User's search query (should already be rewritten if multi-turn)
        workspace_id: Tenant scope — enforced in both Qdrant and BM25
        top_k: Final number of results after fusion
        dense_top_k: Dense retrieval candidate pool size (before fusion)
        bm25_top_k: BM25 retrieval candidate pool size (before fusion)
        document_id: Optional — restrict search to a single document

    Returns:
        Fused and ranked chunk list. Each chunk has score_type='rrf'.
        Falls back to dense-only if BM25 index is not ready.
    """
    logger.info(
        f"Hybrid search: workspace={workspace_id[:8]}, "
        f"query='{query[:60]}'"
    )

    # ── Dense vector search ──────────────────────────────────────────────────
    query_vector = embedder.embed_query(query)
    dense_results = vector_store.search(
        query_vector=query_vector,
        workspace_id=workspace_id,
        top_k=dense_top_k,
        document_id=document_id,
    )
    logger.debug(f"Dense search returned {len(dense_results)} results")

    # ── BM25 sparse search ───────────────────────────────────────────────────
    bm25_results: list[dict] = []
    if bm25_index.is_ready:
        bm25_results = bm25_index.search(
            query=query,
            workspace_id=workspace_id,
            top_k=bm25_top_k,
        )
        # Apply optional document_id filter (BM25 doesn't natively support it)
        if document_id:
            bm25_results = [r for r in bm25_results if r["document_id"] == document_id]
        logger.debug(f"BM25 search returned {len(bm25_results)} results")
    else:
        logger.warning(
            "BM25 index not ready — falling back to dense-only retrieval. "
            "Call POST /workspaces/{id}/search/rebuild-bm25 to build the index."
        )

    # ── Graceful fallback ────────────────────────────────────────────────────
    if not bm25_results:
        logger.info("BM25 returned no results — using dense results only")
        return dense_results[:top_k]

    if not dense_results:
        logger.info("Dense search returned no results — using BM25 results only")
        return bm25_results[:top_k]

    # ── Reciprocal Rank Fusion ───────────────────────────────────────────────
    fused = _reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        top_k=top_k,
    )

    logger.info(
        f"Hybrid search complete: dense={len(dense_results)}, "
        f"bm25={len(bm25_results)}, fused={len(fused)}"
    )
    return fused
