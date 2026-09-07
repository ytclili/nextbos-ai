from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi.testclient import TestClient

from app import main as main_module
from app.api.routes import conversations as conversations_route_module
from app.persistence.postgres.models import ConversationMessage, ConversationThread

app = main_module.app


def make_thread(
    *,
    thread_id: str = "thread-created",
    user_id: str = "user-1",
    title: str | None = "测试会话",
    message_count: int = 0,
    metadata: dict[str, Any] | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> ConversationThread:
    """构造测试用会话 ORM 对象。"""

    now = created_at or datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
    return ConversationThread(
        thread_id=thread_id,
        user_id=user_id,
        title=title,
        status="active",
        last_message_at=None,
        message_count=message_count,
        metadata_json=metadata or {},
        created_at=now,
        updated_at=updated_at or now,
    )


def make_message(
    *,
    message_id: str = "00000000-0000-0000-0000-000000000001",
    thread_id: str = "thread-1",
    user_id: str = "user-1",
    role: str = "user",
    message_type: str = "text",
    content: str | None = "今天吃什么？",
    status: str = "completed",
    metadata: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> ConversationMessage:
    """构造测试用消息 ORM 对象。"""

    return ConversationMessage(
        id=UUID(message_id),
        thread_id=thread_id,
        user_id=user_id,
        role=role,
        type=message_type,
        content=content,
        status=status,
        metadata_json=metadata or {},
        created_at=created_at or datetime(2026, 8, 25, 12, 0, tzinfo=UTC),
    )


def test_create_conversation_endpoint_returns_new_thread(monkeypatch) -> None:
    """创建会话接口应该返回新建的 conversation thread。"""

    calls = []
    created_at = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)

    class FakeConversationRepository:
        def __init__(self, session_factory):
            calls.append(("init", session_factory))

        async def create_thread(self, **kwargs):
            calls.append(("create_thread", kwargs))
            return make_thread(
                user_id=kwargs["user_id"],
                title=kwargs["title"],
                metadata=kwargs["metadata"],
                created_at=created_at,
            )

    monkeypatch.setattr(
        conversations_route_module,
        "ConversationRepository",
        FakeConversationRepository,
    )
    app.state.session_factory = "session-factory"

    client = TestClient(app)
    response = client.post(
        "/api/v1/conversations",
        json={
            "user_id": "user-1",
            "title": "测试会话",
            "metadata": {"kind": "manual"},
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "status": "success",
        "message": "success",
        "data": {
            "thread_id": "thread-created",
            "user_id": "user-1",
            "title": "测试会话",
            "status": "active",
            "message_count": 0,
            "last_message_at": None,
            "metadata": {"kind": "manual"},
            "created_at": "2026-08-25T12:00:00Z",
            "updated_at": "2026-08-25T12:00:00Z",
        },
    }
    assert calls == [
        ("init", "session-factory"),
        (
            "create_thread",
            {
                "user_id": "user-1",
                "title": "测试会话",
                "metadata": {"kind": "manual"},
            },
        ),
    ]


def test_create_conversation_endpoint_defaults_title(monkeypatch) -> None:
    """不传 title 时，接口应该让 repository 使用默认标题。"""

    calls = []
    created_at = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)

    class FakeConversationRepository:
        def __init__(self, session_factory):
            pass

        async def create_thread(self, **kwargs):
            calls.append(kwargs)
            return make_thread(
                user_id=kwargs["user_id"],
                title="新会话",
                created_at=created_at,
            )

    monkeypatch.setattr(
        conversations_route_module,
        "ConversationRepository",
        FakeConversationRepository,
    )
    app.state.session_factory = "session-factory"

    client = TestClient(app)
    response = client.post(
        "/api/v1/conversations",
        json={
            "user_id": "user-1",
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["title"] == "新会话"
    assert calls == [
        {
            "user_id": "user-1",
            "title": None,
            "metadata": {},
        }
    ]


def test_list_conversations_endpoint_returns_user_threads(monkeypatch) -> None:
    """会话列表接口应该返回指定用户的会话列表。"""

    calls = []
    created_at = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)

    class FakeConversationRepository:
        def __init__(self, session_factory):
            calls.append(("init", session_factory))

        async def list_threads(self, **kwargs):
            calls.append(("list_threads", kwargs))
            return [
                make_thread(
                    thread_id="thread-2",
                    user_id=kwargs["user_id"],
                    title="第二个会话",
                    message_count=4,
                    metadata={"source": "web"},
                    created_at=created_at,
                ),
                make_thread(
                    thread_id="thread-1",
                    user_id=kwargs["user_id"],
                    title="第一个会话",
                    message_count=2,
                    created_at=created_at,
                ),
            ]

        async def count_threads(self, **kwargs):
            calls.append(("count_threads", kwargs))
            return 7

    monkeypatch.setattr(
        conversations_route_module,
        "ConversationRepository",
        FakeConversationRepository,
    )
    app.state.session_factory = "session-factory"

    client = TestClient(app)
    response = client.get("/api/v1/conversations", params={"user_id": "user-1", "limit": 2})

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "status": "success",
        "message": "success",
        "data": [
            {
                "thread_id": "thread-2",
                "user_id": "user-1",
                "title": "第二个会话",
                "status": "active",
                "message_count": 4,
                "last_message_at": None,
                "metadata": {"source": "web"},
                "created_at": "2026-08-25T12:00:00Z",
                "updated_at": "2026-08-25T12:00:00Z",
            },
            {
                "thread_id": "thread-1",
                "user_id": "user-1",
                "title": "第一个会话",
                "status": "active",
                "message_count": 2,
                "last_message_at": None,
                "metadata": {},
                "created_at": "2026-08-25T12:00:00Z",
                "updated_at": "2026-08-25T12:00:00Z",
            },
        ],
        "total": 7,
        "limit": 2,
    }
    assert calls == [
        ("init", "session-factory"),
        ("list_threads", {"user_id": "user-1", "limit": 2}),
        ("count_threads", {"user_id": "user-1"}),
    ]


def test_list_conversation_messages_endpoint_returns_thread_history(monkeypatch) -> None:
    """会话历史接口应该按 thread_id 和 user_id 返回消息列表。"""

    calls = []
    created_at = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)

    class FakeConversationRepository:
        def __init__(self, session_factory):
            calls.append(("init", session_factory))

        async def list_messages(self, **kwargs):
            calls.append(("list_messages", kwargs))
            return [
                make_message(
                    message_id="00000000-0000-0000-0000-000000000001",
                    thread_id=kwargs["thread_id"],
                    user_id=kwargs["user_id"],
                    role="user",
                    content="今天吃什么？",
                    created_at=created_at,
                ),
                make_message(
                    message_id="00000000-0000-0000-0000-000000000002",
                    thread_id=kwargs["thread_id"],
                    user_id=kwargs["user_id"],
                    role="assistant",
                    content="可以吃粤菜。",
                    metadata={"type": "text"},
                    created_at=created_at,
                ),
            ]

        async def count_messages(self, **kwargs):
            calls.append(("count_messages", kwargs))
            return 9

    monkeypatch.setattr(
        conversations_route_module,
        "ConversationRepository",
        FakeConversationRepository,
    )
    app.state.session_factory = "session-factory"

    client = TestClient(app)
    response = client.get(
        "/api/v1/conversations/thread-1/messages",
        params={"user_id": "user-1", "limit": 2},
    )

    assert response.status_code == 200
    assert response.json() == {
        "code": 200,
        "status": "success",
        "message": "success",
        "data": [
            {
                "message_id": "00000000-0000-0000-0000-000000000001",
                "thread_id": "thread-1",
                "user_id": "user-1",
                "role": "user",
                "type": "text",
                "content": "今天吃什么？",
                "metadata": {},
                "status": "completed",
                "created_at": "2026-08-25T12:00:00Z",
            },
            {
                "message_id": "00000000-0000-0000-0000-000000000002",
                "thread_id": "thread-1",
                "user_id": "user-1",
                "role": "assistant",
                "type": "text",
                "content": "可以吃粤菜。",
                "metadata": {"type": "text"},
                "status": "completed",
                "created_at": "2026-08-25T12:00:00Z",
            },
        ],
        "total": 9,
        "limit": 2,
    }
    assert calls == [
        ("init", "session-factory"),
        ("list_messages", {"thread_id": "thread-1", "user_id": "user-1", "limit": 2}),
        ("count_messages", {"thread_id": "thread-1", "user_id": "user-1"}),
    ]
