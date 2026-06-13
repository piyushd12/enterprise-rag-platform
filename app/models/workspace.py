import enum

from sqlalchemy import Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, generate_uuid


class MemberRole(str, enum.Enum):
    OWNER = "owner"     # can delete workspace, manage all members
    ADMIN = "admin"     # can upload docs, invite members
    MEMBER = "member"   # can only read and chat


class Workspace(Base, TimestampMixin):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    slug: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)

    members: Mapped[list["WorkspaceMember"]] = relationship(back_populates="workspace", cascade="all, delete-orphan")

    documents: Mapped[list["Document"]] = relationship(
        back_populates="workspace",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"Workspace(id={self.id!r}, name={self.name!r})"


class WorkspaceMember(Base, TimestampMixin):
    __tablename__ = "workspace_members"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MemberRole] = mapped_column(Enum(MemberRole), default=MemberRole.MEMBER, nullable=False)

    workspace: Mapped["Workspace"] = relationship(back_populates="members")
    user: Mapped["User"] = relationship(back_populates="workspace_memberships")