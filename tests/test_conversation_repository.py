from uuid import UUID

import pytest

from app.conversation.repository import (
    MESSAGE_ROLE_ASSISTANT,
    MESSAGE_ROLE_USER,
    MESSAGE_STATUS_COMPLETED,
    MESSAGE_TYPE_TEXT,
    ConversationRepository,
)
from app.persistence.postgres.models import ConversationMessage, ConversationThread


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


class FakeTransactionContext:
    """测试用事务 context。"""

    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class FakeSession:
    """测试用 SQLAlchemy session，只记录新增的 ORM 对象。"""

    def __init__(self) -> None:
        self.added: list[object] = []
        self.scalars_result: FakeScalarResult | None = None
        self.scalar_statements: list[object] = []
        self.scalar_result: object | None = None
        self.single_scalar_statements: list[object] = []

    def begin(self):
        return FakeTransactionContext()

    def add(self, row: object) -> None:
        self.added.append(row)

    async def flush(self) -> None:
        return None

    async def scalars(self, statement: object):
        self.scalar_statements.append(statement)
        return self.scalars_result or FakeScalarResult([])

    async def scalar(self, statement: object):
        self.single_scalar_statements.append(statement)
        return self.scalar_result


class FakeScalarResult:
    """测试用 SQLAlchemy scalars 结果。"""

    def __init__(self, rows: list[object]) -> None:
        self.rows = rows

    def all(self) -> list[object]:
        return self.rows


class FakeSessionContext:
    """测试用 session_factory context。"""

    def __init__(self, session: FakeSession) -> None:
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class FakeSessionFactory:
    """测试用 session_factory，不连接真实数据库。"""

    def __init__(self) -> None:
        self.session = FakeSession()

    def __call__(self):
        return FakeSessionContext(self.session)


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


@pytest.mark.asyncio
async def test_create_thread_adds_empty_conversation_thread() -> None:
    """创建新会话时，只应该写 conversation_threads，不写 message。"""

    session_factory = FakeSessionFactory()
    repository = ConversationRepository(session_factory)

    thread = await repository.create_thread(
        user_id="user-1",
        title="测试会话",
        metadata={"kind": "manual"},
    )

    assert len(session_factory.session.added) == 1
    assert session_factory.session.added[0] is thread
    assert isinstance(thread, ConversationThread)
    assert UUID(thread.thread_id)
    assert thread.user_id == "user-1"
    assert thread.title == "测试会话"
    assert thread.status == "active"
    assert thread.message_count == 0
    assert thread.last_message_at is None
    assert thread.metadata_json == {"kind": "manual"}


@pytest.mark.asyncio
async def test_list_threads_returns_user_threads() -> None:
    """查询用户会话列表时，应该返回 session 查到的会话行。"""

    session_factory = FakeSessionFactory()
    rows = [
        ConversationThread(thread_id="thread-2", user_id="user-1", title="第二个会话"),
        ConversationThread(thread_id="thread-1", user_id="user-1", title="第一个会话"),
    ]
    session_factory.session.scalars_result = FakeScalarResult(rows)
    repository = ConversationRepository(session_factory)

    threads = await repository.list_threads(user_id="user-1", limit=2)

    assert threads == rows
    assert len(session_factory.session.scalar_statements) == 1


@pytest.mark.asyncio
async def test_count_threads_returns_user_thread_total() -> None:
    """统计用户会话总数时，应该返回数据库 count 结果。"""

    session_factory = FakeSessionFactory()
    session_factory.session.scalar_result = 7
    repository = ConversationRepository(session_factory)

    total = await repository.count_threads(user_id="user-1")

    assert total == 7
    assert len(session_factory.session.single_scalar_statements) == 1


@pytest.mark.asyncio
async def test_list_messages_returns_recent_thread_messages_in_chronological_order() -> None:
    """查询会话历史时，应该返回最近 N 条消息，并按旧到新排列。"""

    session_factory = FakeSessionFactory()
    newest_first_rows = [
        ConversationMessage(
            id=UUID("00000000-0000-0000-0000-000000000002"),
            thread_id="thread-1",
            user_id="user-1",
            role="assistant",
            type="text",
            content="可以吃粤菜。",
            status="completed",
            metadata_json={},
        ),
        ConversationMessage(
            id=UUID("00000000-0000-0000-0000-000000000001"),
            thread_id="thread-1",
            user_id="user-1",
            role="user",
            type="text",
            content="今天吃什么？",
            status="completed",
            metadata_json={},
        ),
    ]
    session_factory.session.scalars_result = FakeScalarResult(newest_first_rows)
    repository = ConversationRepository(session_factory)

    messages = await repository.list_messages(thread_id="thread-1", user_id="user-1", limit=2)

    assert messages == list(reversed(newest_first_rows))
    assert len(session_factory.session.scalar_statements) == 1


@pytest.mark.asyncio
async def test_count_messages_returns_thread_message_total() -> None:
    """统计会话历史总数时，应该按 thread_id 和 user_id 过滤。"""

    session_factory = FakeSessionFactory()
    session_factory.session.scalar_result = 9
    repository = ConversationRepository(session_factory)

    total = await repository.count_messages(thread_id="thread-1", user_id="user-1")

    assert total == 9
    assert len(session_factory.session.single_scalar_statements) == 1
