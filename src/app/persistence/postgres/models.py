from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ConversationThread(Base):
    """会话线程表。

    这里保存业务视角的一条会话，而不是 LangGraph 的 checkpoint。
    Redis 里的 checkpoint 过期后，仍然可以通过这张表知道用户有哪些历史会话。
    """

    __tablename__ = "conversation_threads"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)

    # 对外暴露的会话 ID，也就是 chat 接口传入的 thread_id。
    # 这里保持唯一，方便后续按 thread_id 快速找到会话。
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)

    # 用户 ID。长期记忆按 user_id 走，聊天记录也必须能按 user_id 查询。
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # 可选标题，后续前端会话列表可以展示；第一版可以为空。
    title: Mapped[str | None] = mapped_column(String(256))

    # 会话状态。第一版先用字符串，避免过早引入复杂枚举。
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active", index=True)

    # 这个字段用于会话列表排序，不需要扫描 messages 表。
    last_message_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)

    # 冗余消息数，方便列表展示和后续做压缩阈值判断。
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # 预留给前端或业务扩展，例如来源、客户端信息、入口页面等。
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ConversationMessage(Base):
    """会话消息表。

    这里保存人能理解的聊天记录：
    user 问了什么、assistant 回了什么、tool 是否产生过可展示结果。
    不从 Redis checkpoint 反解聊天记录，避免绑定 LangGraph 内部存储结构。
    """

    __tablename__ = "conversation_messages"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)

    # 关联 conversation_threads.thread_id。
    # 使用业务 thread_id，而不是内部 uuid，方便排障时直接用接口参数查询。
    thread_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("conversation_threads.thread_id"),
        nullable=False,
        index=True,
    )

    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # 消息角色：user / assistant / system / tool。
    role: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    # 前端渲染类型：text / card / table / form / action 等。
    # 第一版普通聊天就是 text。
    type: Mapped[str] = mapped_column(String(32), nullable=False, default="text")

    # 消息正文。卡片类消息可以为空，具体渲染数据放 metadata_json。
    content: Mapped[str | None] = mapped_column(Text)

    # 消息状态：completed / failed / cancelled。
    # 后续如果做流式输出，可以扩展 pending / streaming。
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="completed", index=True)

    # 本条消息关联的 trace_id，方便从聊天记录反查 SigNoZ 链路。
    trace_id: Mapped[str | None] = mapped_column(String(64), index=True)

    # 本条 assistant 消息使用的模型配置快照。
    # user 消息通常为空；assistant 消息后续可以绑定具体 snapshot。
    llm_snapshot_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("llm_effective_config_snapshots.id"),
    )

    # 前端展示元数据、tool 结果摘要、usage、错误信息等都先放这里。
    # 注意：不要放明文 API key 或其他 secret。
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ConversationSummary(Base):
    """会话摘要表。

    Redis 里的 summary 是短期运行状态，会随 TTL 过期。
    这张表用于后续把会话摘要长期保存到 PostgreSQL。
    第一版可以先建表，后续到压缩节点或后台任务时再写入。
    """

    __tablename__ = "conversation_summaries"

    id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)

    thread_id: Mapped[str] = mapped_column(
        String(128),
        ForeignKey("conversation_threads.thread_id"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    # 当前滚动摘要文本。
    summary: Mapped[str] = mapped_column(Text, nullable=False)

    # 这份 summary 覆盖到哪条消息。后续追加新消息时，可以从这里之后继续总结。
    covered_through_message_id: Mapped[UUID | None] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("conversation_messages.id"),
    )

    # 摘要覆盖的消息数量，方便判断是否需要重新压缩。
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # 粗略 token 数。第一版可以为空，后续接 tokenizer 后再精确记录。
    token_estimate: Mapped[int | None] = mapped_column(Integer)

    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

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
