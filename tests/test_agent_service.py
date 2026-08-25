from uuid import uuid4

from langchain_core.messages import AIMessage

import app.services.agent_service as agent_service_module
from app.agent.options import ChatModelOptions
from app.core.config import Settings
from app.services.agent_service import AgentService


class FakeConversationRepository:
    """测试用会话仓储。

    不连接真实 PostgreSQL，只记录 AgentService 的写入顺序。
    """

    def __init__(self, session_factory) -> None:
        self.session_factory = session_factory
        self.events: list[tuple] = []
        created_repositories.append(self)

    async def append_user_message(self, **kwargs) -> None:
        """记录用户消息写入。"""

        self.events.append(("user", kwargs))

    async def append_assistant_message(self, **kwargs) -> None:
        """记录 assistant 消息写入。"""

        self.events.append(("assistant", kwargs))


created_repositories: list[FakeConversationRepository] = []


async def test_agent_service_persists_user_and_assistant_messages(monkeypatch) -> None:
    """chat 主链路应该把用户消息和最终回复保存到 PostgreSQL 会话仓储。"""

    events: list[tuple] = []
    snapshot_id = uuid4()

    async def fake_run_graph(*args, **kwargs):
        events.append(("run_graph", kwargs))
        return {
            "messages": [AIMessage(content="可以吃粤菜。")],
            "llm_snapshot_id": snapshot_id,
        }

    created_repositories.clear()
    monkeypatch.setattr(agent_service_module, "ConversationRepository", FakeConversationRepository)
    monkeypatch.setattr(agent_service_module, "run_graph", fake_run_graph)

    service = AgentService(
        checkpointer="checkpointer",
        session_factory="session-factory",
        settings=Settings(),
        memory_store="memory-store",
    )

    content = await service.chat(
        thread_id="thread-1",
        user_id="user-1",
        message="今天吃什么？",
        model_options=ChatModelOptions(),
        trace_id="trace-1",
    )

    assert content == "可以吃粤菜。"
    assert len(created_repositories) == 1

    repository = created_repositories[0]
    assert repository.session_factory == "session-factory"
    assert repository.events == [
        (
            "user",
            {
                "thread_id": "thread-1",
                "user_id": "user-1",
                "content": "今天吃什么？",
                "trace_id": "trace-1",
            },
        ),
        (
            "assistant",
            {
                "thread_id": "thread-1",
                "user_id": "user-1",
                "content": "可以吃粤菜。",
                "trace_id": "trace-1",
                "llm_snapshot_id": snapshot_id,
            },
        ),
    ]
    assert events == [
        (
            "run_graph",
            {
                "thread_id": "thread-1",
                "user_id": "user-1",
                "message": "今天吃什么？",
                "model_options": ChatModelOptions(),
                "session_factory": "session-factory",
                "settings": service.settings,
                "memory_store": "memory-store",
            },
        )
    ]
