from contextlib import asynccontextmanager

from fastapi.testclient import TestClient

from app import main as main_module
from app.api.routes import chat as chat_route_module
from app.core.config import Settings

app = main_module.app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_model_params_validation_rejects_invalid_temperature():
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": "thread-1",
            "user_id": "user-1",
            "message": "今天吃什么？",
            "model_params": {"temperature": 3},
        },
    )

    assert response.status_code == 422


def test_chat_response_schema_contains_frontend_render_items():
    schema = app.openapi()["components"]["schemas"]["ChatResponse"]

    assert "status" in schema["properties"]
    assert "thread_id" in schema["properties"]
    assert "trace_id" in schema["properties"]
    assert "items" in schema["properties"]


def test_chat_response_schema_does_not_expose_internal_observability_fields():
    """chat 响应不应该把后端观测细节直接暴露给前端。"""

    schema = app.openapi()["components"]["schemas"]["ChatResponse"]
    item_schema = app.openapi()["components"]["schemas"]["ChatResponseItem"]

    assert "usage" not in schema["properties"]
    assert "raw" not in schema["properties"]
    assert "usage" not in item_schema["properties"]
    assert "raw" not in item_schema["properties"]


def test_chat_endpoint_passes_trace_id_to_agent_service(monkeypatch):
    """chat 接口应该把当前 trace_id 传给会话落库链路。"""

    calls = []

    class FakeAgentService:
        def __init__(self, checkpointer, session_factory, settings, memory_store=None):
            calls.append(
                (
                    "init",
                    {
                        "checkpointer": checkpointer,
                        "session_factory": session_factory,
                        "settings": settings,
                        "memory_store": memory_store,
                    },
                )
            )

        async def chat(self, **kwargs):
            calls.append(("chat", kwargs))
            return "测试回复"

    settings = Settings()
    monkeypatch.setattr(chat_route_module, "AgentService", FakeAgentService)
    monkeypatch.setattr(chat_route_module, "_current_trace_id", lambda: "trace-123")

    app.state.checkpointer = "checkpointer"
    app.state.session_factory = "session-factory"
    app.state.settings = settings
    app.state.memory_store = "memory-store"

    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": "thread-1",
            "user_id": "user-1",
            "message": "今天吃什么？",
        },
    )

    assert response.status_code == 200
    assert response.json()["trace_id"] == "trace-123"
    assert calls == [
        (
            "init",
            {
                "checkpointer": "checkpointer",
                "session_factory": "session-factory",
                "settings": settings,
                "memory_store": "memory-store",
            },
        ),
        (
            "chat",
            {
                "thread_id": "thread-1",
                "user_id": "user-1",
                "message": "今天吃什么？",
                "model_options": chat_route_module.ChatModelOptions(
                    model_alias=None,
                    model_params=None,
                ),
                "trace_id": "trace-123",
            },
        ),
    ]


def test_chat_stream_endpoint_returns_sse_events(monkeypatch):
    """stream chat 接口应该把 AgentService 的事件编码成 SSE。"""

    calls = []

    class FakeAgentService:
        def __init__(self, checkpointer, session_factory, settings, memory_store=None):
            calls.append(
                (
                    "init",
                    {
                        "checkpointer": checkpointer,
                        "session_factory": session_factory,
                        "settings": settings,
                        "memory_store": memory_store,
                    },
                )
            )

        async def stream_chat(self, **kwargs):
            calls.append(("stream_chat", kwargs))
            yield ("start", {"code": 200, "status": "success", "thread_id": kwargs["thread_id"]})
            yield ("token", {"type": "text", "content": "粤"})
            yield (
                "tool_start",
                {
                    "name": "search_memory",
                    "tool_call_id": "call-1",
                    "message": "正在调用 search_memory",
                },
            )
            yield (
                "tool_end",
                {
                    "name": "search_memory",
                    "tool_call_id": "call-1",
                    "status": "success",
                },
            )
            yield ("token", {"type": "text", "content": "菜"})
            yield ("done", {"content": "粤菜"})

    settings = Settings()
    monkeypatch.setattr(chat_route_module, "AgentService", FakeAgentService)
    monkeypatch.setattr(chat_route_module, "_current_trace_id", lambda: "trace-123")

    app.state.checkpointer = "checkpointer"
    app.state.session_factory = "session-factory"
    app.state.settings = settings
    app.state.memory_store = "memory-store"

    client = TestClient(app)
    with client.stream(
        "POST",
        "/api/v1/chat/stream",
        json={
            "thread_id": "thread-1",
            "user_id": "user-1",
            "message": "今天吃什么？",
        },
    ) as response:
        body = response.read().decode()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: start\ndata: {"code":200,"status":"success","thread_id":"thread-1"}' in body
    assert 'event: token\ndata: {"type":"text","content":"粤"}' in body
    assert (
        'event: tool_start\ndata: {"name":"search_memory","tool_call_id":"call-1",'
        '"message":"正在调用 search_memory"}'
    ) in body
    assert (
        'event: tool_end\ndata: {"name":"search_memory","tool_call_id":"call-1",'
        '"status":"success"}'
    ) in body
    assert 'event: token\ndata: {"type":"text","content":"菜"}' in body
    assert 'event: done\ndata: {"content":"粤菜"}' in body
    assert calls == [
        (
            "init",
            {
                "checkpointer": "checkpointer",
                "session_factory": "session-factory",
                "settings": settings,
                "memory_store": "memory-store",
            },
        ),
        (
            "stream_chat",
            {
                "thread_id": "thread-1",
                "user_id": "user-1",
                "message": "今天吃什么？",
                "model_options": chat_route_module.ChatModelOptions(
                    model_alias=None,
                    model_params=None,
                ),
                "trace_id": "trace-123",
            },
        ),
    ]


def test_chat_endpoint_maps_timeout_to_structured_504(monkeypatch):
    """chat 接口应该把 timeout 映射成前端可识别的结构化错误。"""

    class TimeoutAgentService:
        def __init__(self, checkpointer, session_factory, settings, memory_store=None):
            pass

        async def chat(self, **kwargs):
            raise TimeoutError("Request timed out.")

    monkeypatch.setattr(chat_route_module, "AgentService", TimeoutAgentService)

    app.state.checkpointer = "checkpointer"
    app.state.session_factory = "session-factory"
    app.state.settings = Settings()
    app.state.memory_store = "memory-store"

    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        json={
            "thread_id": "thread-1",
            "user_id": "user-1",
            "message": "写一篇 2000 字作文",
        },
    )

    assert response.status_code == 504
    assert response.json()["detail"] == {
        "code": "llm_timeout",
        "message": "模型请求超时，请稍后重试或缩短输入。",
    }


def test_lifespan_attaches_memory_store(monkeypatch):
    """应用启动时应该把长期记忆 Store 挂到 app.state。"""

    events = []

    class FakeEngine:
        async def dispose(self) -> None:
            events.append("engine_disposed")

    @asynccontextmanager
    async def fake_redis_checkpointer(settings):
        events.append(("redis_started", settings.redis_url))
        yield "redis-checkpointer"

    @asynccontextmanager
    async def fake_postgres_memory_store(settings):
        events.append(("memory_store_started", settings.postgres_dsn))
        yield "memory-store"

    async def fake_initialize_agent_schema(engine) -> None:
        events.append(("schema_initialized", engine))

    settings = Settings()
    engine = FakeEngine()

    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        main_module,
        "configure_logging",
        lambda level: events.append(("logging", level)),
    )
    monkeypatch.setattr(
        main_module,
        "configure_tracing",
        lambda app, settings: events.append("tracing_started"),
    )
    monkeypatch.setattr(main_module, "shutdown_tracing", lambda: events.append("tracing_shutdown"))
    monkeypatch.setattr(main_module, "create_engine", lambda settings: engine)
    monkeypatch.setattr(main_module, "initialize_agent_schema", fake_initialize_agent_schema)
    monkeypatch.setattr(main_module, "create_session_factory", lambda engine: "session-factory")
    monkeypatch.setattr(main_module, "redis_checkpointer", fake_redis_checkpointer)
    monkeypatch.setattr(main_module, "postgres_memory_store", fake_postgres_memory_store)

    with TestClient(app) as client:
        assert client.app.state.settings is settings
        assert client.app.state.db_engine is engine
        assert client.app.state.session_factory == "session-factory"
        assert client.app.state.checkpointer == "redis-checkpointer"
        assert client.app.state.memory_store == "memory-store"

    assert ("logging", settings.log_level) in events
    assert "tracing_started" in events
    assert ("schema_initialized", engine) in events
    assert ("memory_store_started", settings.postgres_dsn) in events
    assert "engine_disposed" in events
    assert "tracing_shutdown" in events
