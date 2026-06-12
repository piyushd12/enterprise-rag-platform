import re

from pydantic import BaseModel, field_validator

from app.models.workspace import MemberRole

class CreateWorkspaceRequest(BaseModel):
    name: str
    description: str | None = None

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Workspace name cannot be empty")
        return v.strip()

    def to_slug(self) -> str:
        """Convert 'My Company Docs' → 'my-company-docs'"""
        slug = self.name.lower().strip()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        return slug.strip("-")

class WorkspaceResponse(BaseModel):
    id: str
    name: str
    slug: str
    description: str | None

    model_config = {"from_attributes": True}

class WorkspaceMemberResponse(BaseModel):
    user_id: str
    role: MemberRole

    model_config = {"from_attributes": True}