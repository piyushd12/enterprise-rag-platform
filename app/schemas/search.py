from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=50)
    document_id: str | None = Field(
        default=None,
        description="Optionally restrict search to a single document"
    )


class ChunkResult(BaseModel):
    chunk_text: str
    chunk_id: str
    document_id: str
    page_num: int
    chunk_index: int
    score: float
    score_type: str | None = Field(
        default=None,
        description="'rrf' for hybrid results, None for dense-only"
    )


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    total_found: int