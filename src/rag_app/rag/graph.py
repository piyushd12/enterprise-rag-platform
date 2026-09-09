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
from rag_app.core.interfaces import EmbeddingProvider, LLMProvider, VectorStore
from rag_app.rag.nodes import (
    _redirect_after_cache_lookup,
    make_cache_lookup_node,
    make_cache_write_node,
    make_generate_node,
    make_retrieve_node,
)
from rag_app.rag.state import RAGState


def build_rag_graph(
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    llm_provider: LLMProvider,
    cache: RedisCache | None = None,
) -> StateGraph:
    """Build and compile the RAG state graph.

    Topology (Phase 2):
        cache_lookup → [hit] END
                      [miss] retrieve → generate → cache_write → END

    Cache behavior:
        - cache_lookup checks Redis for a cached answer (keyed by normalized
          query + collection_version counter). On hit, returns the cached
          answer directly without LLM inference.
        - cache_write stores newly generated answers in Redis with TTL.
        - When cache is None, cache_lookup/cache_write are skipped and
          the graph degrades to the Phase 1 linear flow.

    LangSmith tracing: The compiled graph is invoked with
    config={"metadata": {"request_id": ...}} to auto-trace every
    node as a child run with the request_id for correlation.

    Returns:
        A compiled LangGraph StateGraph.
    """
    # Create node functions
    retrieve = make_retrieve_node(embedding_provider, vector_store)
    generate = make_generate_node(llm_provider)

    # Build the graph
    graph = StateGraph(RAGState)

    # Add compute nodes (always present)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

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
                "retrieve": "retrieve",
            },
        )
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "cache_write")
        graph.add_edge("cache_write", END)
    else:
        # Fallback: Phase 1 linear flow (no caching)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

    # Compile
    return graph.compile()
