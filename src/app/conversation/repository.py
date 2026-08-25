from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Select, desc, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.persistence.postgres.models import (
    ConversationMessage,
    ConversationSummary,
    ConversationThread,
)

MESSAGE_ROLE_USER = "user"
MESSAGE_ROLE_ASSISTANT = "assistant"
MESSAGE_ROLE_SYSTEM = "system"
MESSAGE_ROLE_TOOL = "tool"

MESSAGE_TYPE_TEXT = "text"
MESSAGE_STATUS_COMPLETED = "completed"
THREAD_STATUS_ACTIVE = "active"


class ConversationRepository:
    """会话记录仓储。

    这里保存的是业务聊天记录，不是 LangGraph checkpoint。

    设计重点：
    - Redis checkpoint 继续负责短期运行状态；
    - PostgreSQL conversation_* 表负责长期聊天记录；
    - 每个写方法都自己开启一个很短的事务；
    - 不允许把事务持到 LLM / tool / 外部 HTTP 调用期间。
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self.session_factory = session_factory

    async def append_user_message(
        self,
        *,
        thread_id: str,
        user_id: str,
        content: str,
        trace_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """保存用户消息。

        chat 接口收到请求后可以先调用这个方法。
        它只做短事务写入，提交后马上释放连接。
        """

        return await self.append_message(
            thread_id=thread_id,
            user_id=user_id,
            role=MESSAGE_ROLE_USER,
            content=content,
            message_type=MESSAGE_TYPE_TEXT,
            status=MESSAGE_STATUS_COMPLETED,
            trace_id=trace_id,
            metadata=metadata,
        )

    async def append_assistant_message(
        self,
        *,
        thread_id: str,
        user_id: str,
        content: str | None,
        trace_id: str | None = None,
        llm_snapshot_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """保存 assistant 回复。

        LangGraph 执行完成后调用这个方法。
        第一版先保存最终文本回复，tool 明细后面再决定是否单独落 message。
        """

        return await self.append_message(
            thread_id=thread_id,
            user_id=user_id,
            role=MESSAGE_ROLE_ASSISTANT,
            content=content,
            message_type=MESSAGE_TYPE_TEXT,
            status=MESSAGE_STATUS_COMPLETED,
            trace_id=trace_id,
            llm_snapshot_id=llm_snapshot_id,
            metadata=metadata,
        )

    async def append_message(
        self,
        *,
        thread_id: str,
        user_id: str,
        role: str,
        content: str | None,
        message_type: str = MESSAGE_TYPE_TEXT,
        status: str = MESSAGE_STATUS_COMPLETED,
        trace_id: str | None = None,
        llm_snapshot_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """保存一条会话消息。

        这个方法会在同一个短事务里完成：
        1. 确保 conversation_threads 有对应 thread；
        2. 插入 conversation_messages；
        3. 更新 thread 的 last_message_at 和 message_count。

        注意：这里的事务只包数据库写入，不包 LLM 调用。
        """

        now = datetime.now(UTC)
        metadata = metadata or {}

        async with self.session_factory() as session:
            async with session.begin():
                await self._upsert_thread(
                    session,
                    thread_id=thread_id,
                    user_id=user_id,
                    last_message_at=now,
                )

                message = ConversationMessage(
                    thread_id=thread_id,
                    user_id=user_id,
                    role=role,
                    type=message_type,
                    content=content,
                    status=status,
                    trace_id=trace_id,
                    llm_snapshot_id=llm_snapshot_id,
                    metadata_json=metadata,
                    created_at=now,
                )
                session.add(message)
                await session.flush()

                await session.execute(
                    update(ConversationThread)
                    .where(ConversationThread.thread_id == thread_id)
                    .values(
                        last_message_at=now,
                        message_count=ConversationThread.message_count + 1,
                        updated_at=now,
                    )
                )

                return message.id

    async def save_summary(
        self,
        *,
        thread_id: str,
        user_id: str,
        summary: str,
        covered_through_message_id: UUID | None = None,
        message_count: int = 0,
        token_estimate: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """保存一份会话摘要。

        Redis 里的 summary 会随 checkpoint TTL 过期。
        后续当 summary 节点或后台任务生成稳定摘要时，可以调用这里长期保存。
        """

        now = datetime.now(UTC)
        metadata = metadata or {}

        async with self.session_factory() as session:
            async with session.begin():
                await self._upsert_thread(
                    session,
                    thread_id=thread_id,
                    user_id=user_id,
                    last_message_at=None,
                )

                row = ConversationSummary(
                    thread_id=thread_id,
                    user_id=user_id,
                    summary=summary,
                    covered_through_message_id=covered_through_message_id,
                    message_count=message_count,
                    token_estimate=token_estimate,
                    metadata_json=metadata,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
                await session.flush()
                return row.id

    async def list_recent_messages(
        self,
        *,
        thread_id: str,
        limit: int = 20,
    ) -> list[ConversationMessage]:
        """读取最近 N 条消息。

        这个方法后续用于：
        - Redis checkpoint 过期后，从 PostgreSQL 恢复最近上下文；
        - 后台会话详情页展示；
        - summary 任务读取未总结消息。

        返回顺序是从旧到新，方便直接组装上下文。
        """

        statement: Select[tuple[ConversationMessage]] = (
            select(ConversationMessage)
            .where(ConversationMessage.thread_id == thread_id)
            .order_by(desc(ConversationMessage.created_at))
            .limit(limit)
        )

        async with self.session_factory() as session:
            rows = (await session.scalars(statement)).all()

        return list(reversed(rows))

    async def get_latest_summary(self, *, thread_id: str) -> ConversationSummary | None:
        """读取某个 thread 最新的一份长期摘要。"""

        statement = (
            select(ConversationSummary)
            .where(ConversationSummary.thread_id == thread_id)
            .order_by(desc(ConversationSummary.updated_at))
            .limit(1)
        )

        async with self.session_factory() as session:
            return await session.scalar(statement)

    async def get_thread(self, *, thread_id: str) -> ConversationThread | None:
        """按 thread_id 查询会话线程。"""

        statement = select(ConversationThread).where(
            ConversationThread.thread_id == thread_id,
        )

        async with self.session_factory() as session:
            return await session.scalar(statement)

    @staticmethod
    async def _upsert_thread(
        session: AsyncSession,
        *,
        thread_id: str,
        user_id: str,
        last_message_at: datetime | None,
    ) -> None:
        """确保会话线程存在。

        这里用 PostgreSQL upsert，避免两个请求同时创建同一个 thread 时发生唯一键冲突。
        """

        now = datetime.now(UTC)
        values: dict[str, Any] = {
            "thread_id": thread_id,
            "user_id": user_id,
            "status": THREAD_STATUS_ACTIVE,
            "updated_at": now,
        }
        if last_message_at is not None:
            values["last_message_at"] = last_message_at

        update_values: dict[str, Any] = {
            "user_id": user_id,
            "updated_at": now,
        }
        if last_message_at is not None:
            update_values["last_message_at"] = last_message_at

        statement = (
            insert(ConversationThread)
            .values(**values)
            .on_conflict_do_update(
                index_elements=[ConversationThread.thread_id],
                set_=update_values,
            )
        )

        await session.execute(statement)
