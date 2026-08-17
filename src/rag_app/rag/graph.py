"""
LangGraph RAG state graph definition.

Defines and compiles the RAG pipeline as an explicit state graph:
    retrieve → generate

The graph is designed for extensibility — cache_lookup, rerank, and
cache_write nodes can be inserted without restructuring. LangSmith
tracing is automatically active when LANGSMITH_TRACING=true.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from rag_app.core.interfaces import EmbeddingProvider, LLMProvider, VectorStore
from rag_app.rag.nodes import make_generate_node, make_retrieve_node
from rag_app.rag.state import RAGState


def build_rag_graph(
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    llm_provider: LLMProvider,
) -> StateGraph:
    """Build and compile the RAG state graph.

    Current topology (Phase 1):
        retrieve → generate → END

    Future topology (Phase 2+):
        cache_lookup → (hit?) → END
                       (miss?) → retrieve → rerank → generate → cache_write → END

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

    # Add nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    # Define edges
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # Compile
    return graph.compile()
