from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langmem.short_term import RunningSummary

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

    async def save_summary(self, **kwargs) -> None:
        """记录 summary 写入。"""

        self.events.append(("summary", kwargs))


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
            "context": {
                "running_summary": RunningSummary(
                    summary="用户正在考虑今天吃什么。",
                    summarized_message_ids={"message-1", "message-2"},
                    last_summarized_message_id="message-2",
                )
            },
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
        token="chat-token",
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
        (
            "summary",
            {
                "thread_id": "thread-1",
                "user_id": "user-1",
                "summary": "用户正在考虑今天吃什么。",
                "covered_through_message_id": None,
                "message_count": 2,
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
                "token": "chat-token",
                "model_options": ChatModelOptions(),
                "session_factory": "session-factory",
                "settings": service.settings,
                "memory_store": "memory-store",
            },
        )
    ]


async def test_agent_service_skips_summary_when_graph_does_not_return_running_summary(
    monkeypatch,
) -> None:
    """没有 running_summary 时，chat 主链路不应该写 conversation_summaries。"""

    async def fake_run_graph(*args, **kwargs):
        return {"messages": [AIMessage(content="可以吃粤菜。")]}

    created_repositories.clear()
    monkeypatch.setattr(agent_service_module, "ConversationRepository", FakeConversationRepository)
    monkeypatch.setattr(agent_service_module, "run_graph", fake_run_graph)

    service = AgentService(
        checkpointer="checkpointer",
        session_factory="session-factory",
        settings=Settings(),
    )

    await service.chat(
        thread_id="thread-1",
        user_id="user-1",
        message="今天吃什么？",
    )

    repository = created_repositories[0]
    assert [event[0] for event in repository.events] == ["user", "assistant"]


async def test_agent_service_streams_tokens_and_persists_final_assistant_message(
    monkeypatch,
) -> None:
    """stream_chat 应该边返回 token，结束后保存完整 assistant 回复。"""

    snapshot_id = uuid4()
    events_from_runtime: list[tuple] = []

    async def fake_stream_graph(*args, **kwargs):
        events_from_runtime.append(("stream_graph", kwargs))
        yield agent_service_module.GraphStreamEvent(
            mode="messages",
            data=(AIMessageChunk(content="粤"), {}),
        )
        yield agent_service_module.GraphStreamEvent(
            mode="messages",
            data=(AIMessageChunk(content="菜"), {}),
        )
        yield agent_service_module.GraphStreamEvent(
            mode="final_state",
            data={
                "messages": [AIMessage(content="粤菜")],
                "llm_snapshot_id": snapshot_id,
                "context": {
                    "running_summary": RunningSummary(
                        summary="用户想知道今天吃什么。",
                        summarized_message_ids={"message-1", "message-2"},
                        last_summarized_message_id="message-2",
                    )
                },
            },
        )

    created_repositories.clear()
    monkeypatch.setattr(agent_service_module, "ConversationRepository", FakeConversationRepository)
    monkeypatch.setattr(agent_service_module, "stream_graph", fake_stream_graph)

    service = AgentService(
        checkpointer="checkpointer",
        session_factory="session-factory",
        settings=Settings(),
        memory_store="memory-store",
    )

    events = [
        event
        async for event in service.stream_chat(
            thread_id="thread-1",
            user_id="user-1",
            message="今天吃什么？",
            token="stream-token",
            model_options=ChatModelOptions(),
            trace_id="trace-1",
        )
    ]

    assert events == [
        (
            "start",
            {
                "code": 200,
                "status": "success",
                "thread_id": "thread-1",
                "trace_id": "trace-1",
            },
        ),
        ("token", {"type": "text", "content": "粤"}),
        ("token", {"type": "text", "content": "菜"}),
        ("done", {"content": "粤菜"}),
    ]
    assert created_repositories[0].events == [
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
                "content": "粤菜",
                "trace_id": "trace-1",
                "llm_snapshot_id": snapshot_id,
            },
        ),
        (
            "summary",
            {
                "thread_id": "thread-1",
                "user_id": "user-1",
                "summary": "用户想知道今天吃什么。",
                "covered_through_message_id": None,
                "message_count": 2,
            },
        ),
    ]
    assert events_from_runtime == [
        (
            "stream_graph",
            {
                "thread_id": "thread-1",
                "user_id": "user-1",
                "message": "今天吃什么？",
                "token": "stream-token",
                "model_options": ChatModelOptions(),
                "session_factory": "session-factory",
                "settings": service.settings,
                "memory_store": "memory-store",
            },
        )
    ]


async def test_agent_service_stream_skips_summary_model_tokens(monkeypatch) -> None:
    """stream_chat 不应该把内部 summary 节点的 token 推给前端。"""

    async def fake_stream_graph(*args, **kwargs):
        yield agent_service_module.GraphStreamEvent(
            mode="messages",
            data=(AIMessageChunk(content="内部摘要"), {"langgraph_node": "summarize"}),
        )
        yield agent_service_module.GraphStreamEvent(
            mode="messages",
            data=(AIMessageChunk(content="粤菜"), {"langgraph_node": "respond"}),
        )
        yield agent_service_module.GraphStreamEvent(
            mode="final_state",
            data={"messages": [AIMessage(content="粤菜")]},
        )

    created_repositories.clear()
    monkeypatch.setattr(agent_service_module, "ConversationRepository", FakeConversationRepository)
    monkeypatch.setattr(agent_service_module, "stream_graph", fake_stream_graph)

    service = AgentService(
        checkpointer="checkpointer",
        session_factory="session-factory",
        settings=Settings(),
    )

    events = [
        event
        async for event in service.stream_chat(
            thread_id="thread-1",
            user_id="user-1",
            message="今天吃什么？",
        )
    ]

    assert ("token", {"type": "text", "content": "内部摘要"}) not in events
    assert ("token", {"type": "text", "content": "粤菜"}) in events


async def test_agent_service_stream_emits_tool_status_events(monkeypatch) -> None:
    """stream_chat 应该把工具调用开始/结束转换成前端事件。"""

    async def fake_stream_graph(*args, **kwargs):
        yield agent_service_module.GraphStreamEvent(
            mode="updates",
            data={
                "respond": {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "search_memory",
                                    "args": {"query": "用户喜欢吃什么"},
                                    "id": "call-1",
                                }
                            ],
                        )
                    ]
                }
            },
        )
        yield agent_service_module.GraphStreamEvent(
            mode="updates",
            data={
                "tools": {
                    "messages": [
                        ToolMessage(
                            content="用户喜欢粤菜。",
                            tool_call_id="call-1",
                            name="search_memory",
                        )
                    ]
                }
            },
        )
        yield agent_service_module.GraphStreamEvent(
            mode="final_state",
            data={"messages": [AIMessage(content="你之前说过喜欢粤菜。")]},
        )

    created_repositories.clear()
    monkeypatch.setattr(agent_service_module, "ConversationRepository", FakeConversationRepository)
    monkeypatch.setattr(agent_service_module, "stream_graph", fake_stream_graph)

    service = AgentService(
        checkpointer="checkpointer",
        session_factory="session-factory",
        settings=Settings(),
    )

    events = [
        event
        async for event in service.stream_chat(
            thread_id="thread-1",
            user_id="user-1",
            message="我喜欢吃什么？",
        )
    ]

    assert (
        "tool_start",
        {
            "name": "search_memory",
            "tool_call_id": "call-1",
            "message": "正在调用 search_memory",
        },
    ) in events
    assert (
        "tool_end",
        {
            "name": "search_memory",
            "tool_call_id": "call-1",
            "status": "success",
        },
    ) in events


async def test_agent_service_stream_emits_auth_required_and_pauses_when_graph_interrupts(
    monkeypatch,
) -> None:
    """LangGraph interrupt 时，stream_chat 应该通知前端授权并暂停落库。"""

    class FakeInterrupt:
        id = "interrupt-auth"
        value = {
            "type": "auth_required",
            "name": "get_dashboard",
            "tool_call_id": "call-auth",
            "status": "interrupted",
            "status_code": 401,
            "message": "登录已失效或没有权限，请重新登录后再试。",
        }

    async def fake_stream_graph(*args, **kwargs):
        yield agent_service_module.GraphStreamEvent(
            mode="updates",
            data={"__interrupt__": (FakeInterrupt(),)},
        )

    created_repositories.clear()
    monkeypatch.setattr(agent_service_module, "ConversationRepository", FakeConversationRepository)
    monkeypatch.setattr(agent_service_module, "stream_graph", fake_stream_graph)

    service = AgentService(
        checkpointer="checkpointer",
        session_factory="session-factory",
        settings=Settings(),
    )

    events = [
        event
        async for event in service.stream_chat(
            thread_id="thread-1",
            user_id="user-1",
            message="查一下昨天经营数据",
        )
    ]

    assert (
        "auth_required",
        {
            "type": "auth_required",
            "name": "get_dashboard",
            "tool_call_id": "call-auth",
            "status": "interrupted",
            "status_code": 401,
            "message": "登录已失效或没有权限，请重新登录后再试。",
            "interrupt_id": "interrupt-auth",
        },
    ) in events
    assert ("done", {"content": "", "status": "interrupted"}) in events
    assert [event[0] for event in created_repositories[0].events] == ["user"]


async def test_agent_service_stream_resume_persists_assistant_message(
    monkeypatch,
) -> None:
    """resume 成功后，stream_resume_chat 应该继续输出并保存 assistant 回复。"""

    async def fake_stream_graph(*args, **kwargs):
        yield agent_service_module.GraphStreamEvent(
            mode="messages",
            data=(AIMessageChunk(content="昨天应收 100 元。"), {"langgraph_node": "final_respond"}),
        )
        yield agent_service_module.GraphStreamEvent(
            mode="final_state",
            data={"messages": [AIMessage(content="昨天应收 100 元。")]},
        )

    created_repositories.clear()
    monkeypatch.setattr(agent_service_module, "ConversationRepository", FakeConversationRepository)
    monkeypatch.setattr(agent_service_module, "stream_graph_resume", fake_stream_graph)

    service = AgentService(
        checkpointer="checkpointer",
        session_factory="session-factory",
        settings=Settings(),
    )

    events = [
        event
        async for event in service.stream_resume_chat(
            thread_id="thread-1",
            user_id="user-1",
            resume={"type": "auth_result", "status": "success"},
        )
    ]

    assert events == [
        (
            "start",
            {
                "code": 200,
                "status": "success",
                "thread_id": "thread-1",
                "trace_id": None,
            },
        ),
        ("token", {"type": "text", "content": "昨天应收 100 元。"}),
        ("done", {"content": "昨天应收 100 元。"}),
    ]
    assert created_repositories[0].events == [
        (
            "assistant",
            {
                "thread_id": "thread-1",
                "user_id": "user-1",
                "content": "昨天应收 100 元。",
                "trace_id": None,
                "llm_snapshot_id": None,
            },
        )
    ]
