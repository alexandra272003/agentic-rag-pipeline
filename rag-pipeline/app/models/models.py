from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON, CheckConstraint, DateTime, ForeignKey, Index, String, Text, UniqueConstraint, text, true,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.core.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Tenant(Base):
    """An isolation boundary: one customer organisation. Everything a
    tenant owns (users, documents, chunks, audit rows) carries tenant_id."""

    __tablename__ = "tenants"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    slug: Mapped[str] = mapped_column(String(60), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class User(Base):
    """A person, belonging to exactly one tenant, with one role there.
    (A user in several tenants would need a memberships table -- a
    deliberate simplification, noted in the README.)"""

    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('owner','admin','editor','viewer')", name="ck_users_role"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"), index=True)
    email: Mapped[str] = mapped_column(String(254), unique=True)  # stored lowercased
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(10))
    # Deactivation, not deletion: audit rows and document ownership keep
    # pointing at a real user. Checked on EVERY request, so it takes effect
    # immediately even for tokens that haven't expired.
    is_active: Mapped[bool] = mapped_column(default=True, server_default=true())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        # The same file may be ingested more than once with DIFFERENT chunk
        # settings (the Day 33 experiment); the exact same file + settings
        # twice is a duplicate -- but only WITHIN a tenant (a global rule
        # would tell tenant B that tenant A already uploaded this file) and
        # only among live rows (so a soft-deleted document can be re-uploaded).
        Index(
            "uq_doc_active_content", "tenant_id", "content_sha256", "chunk_size", "chunk_overlap",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
        Index("ix_documents_tenant_id_id", "tenant_id", "id"),
        CheckConstraint("visibility IN ('tenant','private')", name="ck_documents_visibility"),
        CheckConstraint("version >= 1", name="ck_documents_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    # Creator. Nullable for documents that predate tenancy (backfilled into
    # the default tenant with no owner) and for system/offline ingestion.
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    filename: Mapped[str] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(10))  # pdf | txt | md
    content_sha256: Mapped[str] = mapped_column(String(64), index=True)
    page_count: Mapped[int]
    chunk_size: Mapped[int]
    chunk_overlap: Mapped[int]
    chunk_count: Mapped[int]
    # 'tenant' = every member can read; 'private' = owner, admins and owners.
    visibility: Mapped[str] = mapped_column(String(10), default="tenant", server_default="tenant")
    # Optimistic-lock counter: bumped on every change; updates must quote
    # the version they read (see document_repository.update_document).
    version: Mapped[int] = mapped_column(default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    updated_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    # Soft delete: the row (and its chunks) stay for audit/restore, but every
    # read path filters deleted_at IS NULL via AccessFilter.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    deleted_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Chunk.chunk_index"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_position"),
        Index("ix_chunks_tenant_document", "tenant_id", "document_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    # Denormalised from the parent document on purpose: lets retrieval and
    # row-level security filter chunks by tenant WITHOUT joining documents,
    # and makes tenant isolation a property of the chunk row itself.
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_index: Mapped[int]  # position within the document, 0-based
    page_number: Mapped[int | None] = mapped_column(nullable=True)
    content: Mapped[str] = mapped_column(Text)
    # Offsets are into the CLEANED text of this chunk's page (or of the whole
    # document for txt/md), so content == cleaned_page_text[char_start:char_end].
    char_start: Mapped[int]
    char_end: Mapped[int]
    token_count: Mapped[int]
    # Day 29: one embedding per chunk, generated locally at ingestion time.
    # Postgres: a real pgvector column (ivfflat cosine index). SQLite (test
    # suite only): a JSON array fallback -- same data, different storage.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(settings.embedding_dimensions).with_variant(JSON(), "sqlite"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped["Document"] = relationship(back_populates="chunks")


class AuditLog(Base):
    """Append-only record of who did what, in which tenant. Written in the
    SAME transaction as the change it describes, so an action and its audit
    row commit or roll back together."""

    __tablename__ = "audit_log"
    __table_args__ = (Index("ix_audit_tenant_created", "tenant_id", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(40))  # e.g. document.created
    resource_type: Mapped[str] = mapped_column(String(30))
    resource_id: Mapped[int | None] = mapped_column(nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
