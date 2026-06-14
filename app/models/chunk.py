from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, generate_uuid


class DocumentChunk(Base, TimestampMixin):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        String(100), primary_key=True, default=generate_uuid
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,       # 
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    page_num: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    char_start: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    char_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    qdrant_point_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return (
            f"DocumentChunk(id={self.id!r}, "
            f"doc={self.document_id!r}, "
            f"index={self.chunk_index})"
        )