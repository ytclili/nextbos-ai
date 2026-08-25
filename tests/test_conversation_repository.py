from uuid import UUID

import pytest

from app.conversation.repository import (
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_USER,
    MESSAGE_STATUS_COMPLETED,
    MESSAGE_TYPE_TEXT,
    ConversationRepository,
)


class RecordingConversationRepository(ConversationRepository):
    """测试用会话仓储。

    这里不连接真实 PostgreSQL，只记录 append_message 收到的参数。
    这样可以验证 public 方法的参数规整逻辑，又不会污染本地数据库。
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def append_message(self, **kwargs) -> UUID:
        """记录待写入消息，并返回固定消息 ID。"""

        self.calls.append(kwargs)
        return UUID("00000000-0000-0000-0000-000000000001")


@pytest.mark.asyncio
async def test_append_user_message_delegates_to_append_message() -> None:
    """保存用户消息时，应该统一转成 user/text/completed 消息。"""

    repository = RecordingConversationRepository()

    message_id = await repository.append_user_message(
        thread_id="thread-1",
        user_id="user-1",
        content="今天吃什么？",
        trace_id="trace-1",
        metadata={"source": "test"},
    )

    assert message_id == UUID("00000000-0000-0000-0000-000000000001")
    assert repository.calls == [
        {
            "thread_id": "thread-1",
            "user_id": "user-1",
            "role": MESSAGE_ROLE_USER,
            "content": "今天吃什么？",
            "message_type": MESSAGE_TYPE_TEXT,
            "status": MESSAGE_STATUS_COMPLETED,
            "trace_id": "trace-1",
            "metadata": {"source": "test"},
        }
    ]


@pytest.mark.asyncio
async def test_append_assistant_message_delegates_to_append_message() -> None:
    """保存 assistant 回复时，应该保留 trace 和模型快照引用。"""

    repository = RecordingConversationRepository()
    snapshot_id = UUID("00000000-0000-0000-0000-000000000002")

    message_id = await repository.append_assistant_message(
        thread_id="thread-1",
        user_id="user-1",
        content="可以吃粤菜。",
        trace_id="trace-2",
        llm_snapshot_id=snapshot_id,
        metadata={"type": "text"},
    )

    assert message_id == UUID("00000000-0000-0000-0000-000000000001")
    assert repository.calls == [
        {
            "thread_id": "thread-1",
            "user_id": "user-1",
            "role": MESSAGE_ROLE_ASSISTANT,
            "content": "可以吃粤菜。",
            "message_type": MESSAGE_TYPE_TEXT,
            "status": MESSAGE_STATUS_COMPLETED,
            "trace_id": "trace-2",
            "llm_snapshot_id": snapshot_id,
            "metadata": {"type": "text"},
        }
    ]
