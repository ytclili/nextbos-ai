from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class MemoryRecord(Base):
    __tablename__ = "agent_memories"
    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    namespace: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    memory_type: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    importance: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    source_thread_id: Mapped[str | None] = mapped_column(String(128))
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class LLMProviderCredential(Base):
    """LLM 供应商凭证表。

    这里存加密后的 key 或外部 secret 引用；不要把明文 key 放进模型档案或快照。
    """

    __tablename__ = "llm_provider_credentials"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    encrypted_api_key: Mapped[str | None] = mapped_column(Text)
    secret_ref: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class LLMModelProfile(Base):
    """后台可管理的模型档案表。

    它描述“用哪个 provider、哪个 base_url、哪个模型名、哪组默认参数和哪个凭证”。
    第一版不放 capabilities，避免没人维护的能力字段污染配置。
    """

    __tablename__ = "llm_model_profiles"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    alias: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    base_url: Mapped[str] = mapped_column(String(512), nullable=False)
    credential_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("llm_provider_credentials.id")
    )
    default_params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active", index=True)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    scope: Mapped[str] = mapped_column(String(32), nullable=False, default="global", index=True)
    scope_id: Mapped[str | None] = mapped_column(String(128), index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class LLMEffectiveConfigSnapshot(Base):
    """一次运行最终模型配置的不可变快照表。

    agent run 开始时固化这份配置；后续 resume/replay 应该读快照，而不是重新读 live 默认配置。
    """

    __tablename__ = "llm_effective_config_snapshots"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    model_profile_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("llm_model_profiles.id")
    )
    model_profile_version: Mapped[int | None] = mapped_column(Integer)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    base_url: Mapped[str] = mapped_column(String(512), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    credential_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("llm_provider_credentials.id")
    )
    config_digest: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
