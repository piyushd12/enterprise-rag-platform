"""
Application settings loaded from environment variables via pydantic-settings.
All configuration is centralized here — no module reads env vars directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated configuration for the entire RAG application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM Providers ---
    groq_api_key: str = ""
    openrouter_api_key: str = ""

    # --- Observability ---
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-app"
    langsmith_tracing: bool = True
    logfire_token: str = ""

    # --- Infrastructure ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    redis_url: str = "redis://localhost:6380/0"

    # --- Embedding Model ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- LLM Models ---
    llm_primary_model: str = "llama-3.3-70b-versatile"
    llm_fallback_model: str = "nvidia/llama-3.1-nemotron-70b-instruct:free"

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Caching ---
    cache_ttl_answers: int = 3600
    cache_ttl_embeddings: int = 86400

    # --- Qdrant ---
    qdrant_collection_name: str = "documents"


# Singleton instance — import this wherever config is needed
settings = Settings()
