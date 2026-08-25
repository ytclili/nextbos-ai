from uuid import UUID

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.conversation.context_loader import ConversationContextLoader
from app.conversation.repository import (
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_SYSTEM,
    MESSAGE_ROLE_TOOL,
    MESSAGE_ROLE_USER,
)
from app.persistence.postgres.models import ConversationMessage, ConversationSummary


class FakeConversationRepository:
    """测试用会话仓储。

    不连接真实 PostgreSQL，只返回预先准备好的 summary 和 messages。
    """

    def __init__(
        self,
        *,
        summary: ConversationSummary | None = None,
        messages: list[ConversationMessage] | None = None,
    ) -> None:
        self.summary = summary
        self.messages = messages or []
        self.summary_thread_id: str | None = None
        self.messages_thread_id: str | None = None
        self.messages_limit: int | None = None

    async def get_latest_summary(self, *, thread_id: str) -> ConversationSummary | None:
        """记录 summary 查询参数，并返回测试数据。"""

        self.summary_thread_id = thread_id
        return self.summary

    async def list_recent_messages(
        self,
        *,
        thread_id: str,
        limit: int = 20,
    ) -> list[ConversationMessage]:
        """记录最近消息查询参数，并返回测试数据。"""

        self.messages_thread_id = thread_id
        self.messages_limit = limit
        return self.messages


def make_message(
    *,
    message_id: str,
    role: str,
    content: str | None,
    metadata: dict | None = None,
) -> ConversationMessage:
    """创建一条不落库的 ConversationMessage ORM 对象。"""

    return ConversationMessage(
        id=UUID(message_id),
        thread_id="thread-1",
        user_id="user-1",
        role=role,
        type="text",
        content=content,
        status="completed",
        metadata_json=metadata or {},
    )


@pytest.mark.asyncio
async def test_load_messages_returns_summary_then_recent_messages() -> None:
    """恢复上下文时，应该先放长期摘要，再按时间顺序放最近聊天消息。"""

    repository = FakeConversationRepository(
        summary=ConversationSummary(
            summary="用户喜欢粤菜，最近在讨论午饭选择。",
            thread_id="thread-1",
            user_id="user-1",
        ),
        messages=[
            make_message(
                message_id="00000000-0000-0000-0000-000000000001",
                role=MESSAGE_ROLE_USER,
                content="今天吃什么？",
            ),
            make_message(
                message_id="00000000-0000-0000-0000-000000000002",
                role=MESSAGE_ROLE_ASSISTANT,
                content="可以吃粤菜。",
            ),
        ],
    )
    loader = ConversationContextLoader(repository, recent_message_limit=12)

    messages = await loader.load_messages(thread_id="thread-1")

    assert repository.summary_thread_id == "thread-1"
    assert repository.messages_thread_id == "thread-1"
    assert repository.messages_limit == 12
    assert len(messages) == 3
    assert isinstance(messages[0], SystemMessage)
    assert "用户喜欢粤菜" in str(messages[0].content)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "今天吃什么？"
    assert messages[1].id == "00000000-0000-0000-0000-000000000001"
    assert isinstance(messages[2], AIMessage)
    assert messages[2].content == "可以吃粤菜。"
    assert messages[2].id == "00000000-0000-0000-0000-000000000002"


@pytest.mark.asyncio
async def test_load_messages_skips_blank_summary_and_unknown_role() -> None:
    """空摘要和未知 role 不应该污染模型输入。"""

    repository = FakeConversationRepository(
        summary=ConversationSummary(
            summary="   ",
            thread_id="thread-1",
            user_id="user-1",
        ),
        messages=[
            make_message(
                message_id="00000000-0000-0000-0000-000000000003",
                role="unknown",
                content="这条不应该进入上下文",
            ),
            make_message(
                message_id="00000000-0000-0000-0000-000000000004",
                role=MESSAGE_ROLE_SYSTEM,
                content="你是一个认真回答问题的助手。",
            ),
        ],
    )
    loader = ConversationContextLoader(repository)

    messages = await loader.load_messages(thread_id="thread-1")

    assert len(messages) == 1
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "你是一个认真回答问题的助手。"


@pytest.mark.asyncio
async def test_load_messages_converts_tool_message_with_tool_call_id() -> None:
    """tool 消息应该转换成 LangChain ToolMessage，并保留 tool_call_id。"""

    repository = FakeConversationRepository(
        messages=[
            make_message(
                message_id="00000000-0000-0000-0000-000000000005",
                role=MESSAGE_ROLE_TOOL,
                content="工具执行结果",
                metadata={"tool_call_id": "call-1"},
            ),
        ],
    )
    loader = ConversationContextLoader(repository)

    messages = await loader.load_messages(thread_id="thread-1")

    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].content == "工具执行结果"
    assert messages[0].tool_call_id == "call-1"
    assert messages[0].id == "00000000-0000-0000-0000-000000000005"
