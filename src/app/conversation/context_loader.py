from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.conversation.repository import (
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_SYSTEM,
    MESSAGE_ROLE_TOOL,
    MESSAGE_ROLE_USER,
    ConversationRepository,
)
from app.persistence.postgres.models import ConversationMessage, ConversationSummary

DEFAULT_RECENT_MESSAGE_LIMIT = 20


class ConversationContextLoader:
    """从 PostgreSQL 会话记录恢复 LangGraph 输入上下文。

    Redis checkpoint 是短期运行状态，会因为 TTL 过期而消失。
    conversation_* 表是长期业务聊天记录，所以当 Redis 没有 checkpoint 时，
    runtime 可以用这个 loader 从 PostgreSQL 重新组装一份 LangChain messages。
    """

    def __init__(
        self,
        repository: ConversationRepository,
        *,
        recent_message_limit: int = DEFAULT_RECENT_MESSAGE_LIMIT,
    ) -> None:
        self.repository = repository
        self.recent_message_limit = recent_message_limit

    async def load_messages(self, *, thread_id: str) -> list[AnyMessage]:
        """加载某个 thread 的可恢复上下文。

        返回顺序：
        1. 最新长期摘要，转成 SystemMessage；
        2. 最近 N 条原始聊天消息，按时间从旧到新排列。

        注意：
        - AgentService 当前会先把本轮用户消息写入 PostgreSQL，再调用 run_graph；
        - 所以后续 runtime 接入这个 loader 时，不要再额外重复 append 当前用户消息。
        """

        latest_summary = await self.repository.get_latest_summary(thread_id=thread_id)
        recent_messages = await self.repository.list_recent_messages(
            thread_id=thread_id,
            limit=self.recent_message_limit,
        )

        messages: list[AnyMessage] = []
        summary_message = self._summary_to_message(latest_summary)
        if summary_message is not None:
            messages.append(summary_message)

        messages.extend(
            message
            for row in recent_messages
            if (message := self._conversation_message_to_langchain_message(row)) is not None
        )
        return messages

    @staticmethod
    def _summary_to_message(summary: ConversationSummary | None) -> SystemMessage | None:
        """把长期摘要转换成 system 上下文。"""

        if summary is None or not summary.summary.strip():
            return None

        return SystemMessage(
            content=(
                "以下是这段会话此前的长期摘要，用于在短期 Redis checkpoint 过期后恢复上下文：\n"
                f"{summary.summary}"
            )
        )

    @staticmethod
    def _conversation_message_to_langchain_message(
        row: ConversationMessage,
    ) -> AnyMessage | None:
        """把业务消息行转换成 LangChain 消息对象。"""

        content = row.content or ""

        if row.role == MESSAGE_ROLE_USER:
            return HumanMessage(content=content, id=str(row.id))

        if row.role == MESSAGE_ROLE_ASSISTANT:
            return AIMessage(content=content, id=str(row.id))

        if row.role == MESSAGE_ROLE_SYSTEM:
            return SystemMessage(content=content, id=str(row.id))

        if row.role == MESSAGE_ROLE_TOOL:
            metadata = row.metadata_json or {}
            tool_call_id = metadata.get("tool_call_id") or str(row.id)
            return ToolMessage(
                content=content,
                tool_call_id=str(tool_call_id),
                id=str(row.id),
            )

        # 未知 role 不塞进模型上下文，避免污染 LangGraph 输入。
        return None
