"""
Application settings loaded from environment variables via pydantic-settings.
All configuration is centralized here — no module reads env vars directly.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated configuration for the entire RAG application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # Lets an optional setting be switched off from the environment,
        # e.g. GUARD_RELEVANCE_MIN_SCORE=none.
        env_parse_none_str="none",
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
    llm_primary_model: str = "openai/gpt-oss-20b"
    llm_fallback_model: str = "openrouter/free"

    # --- Chunking ---
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- Caching ---
    cache_ttl_answers: int = 3600
    cache_ttl_embeddings: int = 86400

    # --- Qdrant ---
    qdrant_collection_name: str = "documents"

    # --- Ingestion ---
    # Shared between the API process and the Celery worker (which may run in a
    # separate container) so uploaded files enqueued for async ingestion are
    # visible to whichever process actually processes them. Resolved relative
    # to the process's own working directory — mount this same relative path
    # into the worker container (see docker-compose.yml) rather than passing
    # absolute host paths across the process boundary.
    ingest_tmp_dir: str = "data/temp"

    # --- Guardrails ---
    # Master switch for the detection guards (injection, PII, relevance,
    # output screening). The structural defenses -- the filters allowlist
    # and system/data prompt separation -- are always on, since turning
    # them off would only reopen holes, never fix a false positive.
    guardrails_enabled: bool = True
    # Weighted injection-phrase score at which a query is blocked; any
    # match scoring below this is a flag (logged, not cached) instead.
    guard_injection_block_score: float = 1.0
    # Minimum top cross-encoder score (a raw logit, unbounded and possibly
    # negative) for generation to run at all; below it the request is
    # refused without an LLM call. None disables the floor. Calibrated
    # against data/eval -- see the README's Guardrails section.
    guard_relevance_min_score: float | None = None
    # What happens to retrieved chunks flagged as containing injection
    # text: dropped from the context, or kept (still wrapped as data).
    guard_flagged_chunk_action: Literal["drop", "wrap"] = "drop"
    # What ingestion does with a chunk containing a secret (API key,
    # private key, token): mask it before embedding/storing, keep it but
    # flag it, or skip the chunk.
    guard_ingest_secret_action: Literal["mask", "flag", "drop"] = "mask"
    # Mask PII (emails, phones, cards, national IDs) in answers. Secrets
    # are always masked regardless.
    guard_mask_output_pii: bool = True
    # Flag answers that cite no [Source N] or cite one that doesn't exist.
    guard_citation_check: bool = True
    # Optional LLM-judged groundedness check -- one extra LLM call per
    # answered request, so off by default on the free-tier token budget.
    guard_groundedness_llm: bool = False
    # Classifier-type checks (the groundedness judge) that error or time
    # out either pass ("open", logged as a warning) or return their
    # non-pass outcome ("closed").
    guard_classifier_timeout_s: float = 5.0
    guard_classifier_fail_mode: Literal["open", "closed"] = "open"


# Singleton instance — import this wherever config is needed
settings = Settings()
