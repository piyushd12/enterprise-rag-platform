from pydantic import BaseModel
from app.models.document import DocumentStatus

class DocumentResponse(BaseModel):
    id: str
    workspace_id: str
    title: str
    file_name: str
    file_type: str
    file_size: int
    status: DocumentStatus
    error_message: str | None = None
    page_count: int | None = None
    # Note: extracted_text can be too large for list API responses so not included

    model_config = {"from_attributes": True}