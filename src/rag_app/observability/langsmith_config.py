"""
LangSmith configuration and helpers.

LangSmith handles ALL LLM-pipeline observability:
- LangGraph node-level traces (retrieve, generate, cache-lookup, rerank)
- Prompt/completion pairs, token counts, latencies per node
- LLM provider metadata (which provider served the request)

LangSmith is configured entirely via environment variables:
- LANGSMITH_API_KEY
- LANGSMITH_PROJECT
- LANGSMITH_TRACING=true

No explicit SDK initialization is needed — LangChain/LangGraph auto-detect
these env vars. This module provides helper utilities.
"""

import os
from rag_app.config.settings import settings


def configure_langsmith() -> None:
    """Ensure LangSmith env vars are set from our centralized settings.

    LangChain reads LANGSMITH_API_KEY, LANGSMITH_PROJECT, and
    LANGSMITH_TRACING directly from the environment. This function
    propagates our pydantic-settings values into os.environ so that
    LangChain picks them up regardless of how the app was started.
    """
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    if settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
