"""
LangGraph RAG state graph definition.

Defines and compiles the RAG pipeline as an explicit state graph:

    cache_lookup → [hit?] → END
                   [miss] → retrieve → generate → cache_write → END

The graph is designed for extensibility — rerank and HyDE nodes can be
inserted between retrieve and generate without restructuring. LangSmith
tracing is automatically active when LANGSMITH_TRACING=true.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag_app.caching.redis_cache import RedisCache
from rag_app.core.interfaces import (
    EmbeddingProvider,
    KeywordSearchProvider,
    LLMProvider,
    Reranker,
    VectorStore,
)
from rag_app.rag.nodes import (
    _redirect_after_cache_lookup,
    make_cache_lookup_node,
    make_cache_write_node,
    make_generate_node,
    make_hyde_expand_node,
    make_rerank_node,
    make_retrieve_node,
)
from rag_app.rag.state import RAGState


def build_rag_graph(
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    llm_provider: LLMProvider,
    cache: RedisCache | None = None,
    use_hyde: bool = False,
    reranker: Reranker | None = None,
    keyword_search: KeywordSearchProvider | None = None,
) -> StateGraph:
    """Build and compile the RAG state graph.

    Topology (Phase 2):
        cache_lookup → [hit] END
                      [miss] retrieve → generate → cache_write → END

    Topology (Phase 3, use_hyde=True):
        cache_lookup → [hit] END
                      [miss] hyde_expand → retrieve → generate → cache_write → END

    Cache behavior:
        - cache_lookup checks Redis for a cached answer (keyed by normalized
          query + collection_version counter). On hit, returns the cached
          answer directly without LLM inference.
        - cache_write stores newly generated answers in Redis with TTL.
        - When cache is None, cache_lookup/cache_write are skipped and
          the graph degrades to the Phase 1 linear flow.

    HyDE behavior (optional, off by default):
        - hyde_expand generates a hypothetical answer passage from the raw
          query and stores it as hyde_document; retrieve embeds that
          passage instead of the raw query when present. Costs one extra
          LLM call per cache-miss request, so it is opt-in rather than
          always in the graph.

    Reranker behavior (optional, off when reranker is None):
        - retrieve widens its candidate pool (see RERANK_CANDIDATE_MULTIPLIER
          in nodes.py) instead of searching for exactly top_k; the rerank
          node re-scores that wider pool with a cross-encoder and trims back
          down to top_k before generate ever sees it. Generation's context
          size is unaffected by how wide the candidate pool was -- only the
          intermediate retrieval/rerank step gets wider. Purely local
          compute (no API calls), so unlike HyDE this has no quota cost.

    Keyword search behavior (optional, requires reranker to have any
    effect -- otherwise nothing re-scores the merged set):
        - retrieve also runs BM25 search and merges it with the dense
          candidate pool (deduped) before reranking. Catches passages with
          distinctive exact terms (proper nouns, specific phrases) that
          dense embeddings can under-weight.

    LangSmith tracing: The compiled graph is invoked with
    config={"metadata": {"request_id": ...}} to auto-trace every
    node as a child run with the request_id for correlation.

    Returns:
        A compiled LangGraph StateGraph.
    """
    # Create node functions
    retrieve = make_retrieve_node(
        embedding_provider,
        vector_store,
        use_reranker=reranker is not None,
        keyword_search=keyword_search,
    )
    generate = make_generate_node(llm_provider)

    # Build the graph
    graph = StateGraph(RAGState)

    # Add compute nodes (always present)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    if use_hyde:
        graph.add_node("hyde_expand", make_hyde_expand_node(llm_provider))
        pre_retrieve = "hyde_expand"
        graph.add_edge("hyde_expand", "retrieve")
    else:
        pre_retrieve = "retrieve"

    if reranker is not None:
        graph.add_node("rerank", make_rerank_node(reranker))
        graph.add_edge("retrieve", "rerank")
        post_retrieve = "rerank"
    else:
        post_retrieve = "retrieve"

    # Add cache nodes when cache is provided
    if cache is not None:
        cache_lookup = make_cache_lookup_node(cache)
        cache_write = make_cache_write_node(cache)

        graph.add_node("cache_lookup", cache_lookup)
        graph.add_node("cache_write", cache_write)

        # Edges: cache_lookup branches, cache_write runs after generate
        graph.set_entry_point("cache_lookup")
        graph.add_conditional_edges(
            "cache_lookup",
            _redirect_after_cache_lookup,
            {
                "end": END,
                "retrieve": pre_retrieve,
            },
        )
        graph.add_edge(post_retrieve, "generate")
        graph.add_edge("generate", "cache_write")
        graph.add_edge("cache_write", END)
    else:
        # Fallback: Phase 1 linear flow (no caching)
        graph.set_entry_point(pre_retrieve)
        graph.add_edge(post_retrieve, "generate")
        graph.add_edge("generate", END)

    # Compile
    return graph.compile()
