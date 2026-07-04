from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = "Enterprise RAG Platform"
    debug: bool = False

    database_url: str = "postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb"
    postgres_password: str = "ragpassword"

    redis_url: str = "redis://localhost:6379"

    qdrant_url: str = "http://localhost:6333"

    s3_endpoint_url: str | None = None
    s3_bucket_name: str = "rag-documents"
    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin123"

    secret_key: str = "dev-secret-key-change-in-production"
    jwt_secret_key: str = "dev-jwt-secret-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24

    llm_provider: str = "groq"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    openai_api_key: str | None = None
    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    max_response_tokens: int = 1024
    rag_context_token_budget: int = 4000

@lru_cache
def get_settings() -> Settings:
    return Settings()