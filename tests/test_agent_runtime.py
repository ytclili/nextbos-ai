import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import runtime as runtime_module
from app.agent.options import ChatModelOptions
from app.core.config import Settings
from app.llm.models import EffectiveModelConfig, ProviderCredential


class FakeSessionContext:
    """测试用数据库 session context，不连接真实 PostgreSQL。"""

    async def __aenter__(self):
        return object()

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class FakeSessionFactory:
    """测试用 session_factory。"""

    def __call__(self):
        return FakeSessionContext()


class FakeCheckpointer:
    """测试用 checkpointer。

    checkpoint 为 None 表示 Redis 里没有短期 checkpoint；
    checkpoint 非 None 表示 Redis 里还能找到历史运行状态。
    """

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.config = None

    async def aget_tuple(self, config):
        """记录 LangGraph config，并返回预设 checkpoint。"""

        self.config = config
        return self.checkpoint


@pytest.mark.asyncio
async def test_run_graph_passes_memory_store_and_langgraph_user_id(monkeypatch) -> None:
    """runtime 应该把长期记忆 Store 和用户命名空间配置传给 LangGraph。"""

    captured = {}
    checkpointer = FakeCheckpointer(checkpoint=object())
    memory_store = object()
    summarization_model = object()

    class FakeAgentModelRuntime:
        """测试用模型运行时，不访问真实数据库或大模型。"""

        def __init__(self, *, config_resolver):
            self.config_resolver = config_resolver

        async def resolve_config(self, options):
            captured["model_options"] = options
            return EffectiveModelConfig(
                source="env_fallback",
                provider="openai_compatible",
                base_url="https://api.example.com/v1",
                model_name="test-model",
                params={},
                credential=ProviderCredential(
                    id=None,
                    provider="openai_compatible",
                    name="env",
                    api_key="test-key",
                ),
                digest="digest",
            )

        def create_chat_model(self, config):
            captured["summary_model_config"] = config
            return summarization_model

    class FakeRunnable:
        async def ainvoke(self, state, config):
            captured["state"] = state
            captured["config"] = config
            return {"messages": [AIMessage(content="ok")]}

    def fake_build_graph(
        *,
        checkpointer,
        model_runtime,
        store,
        summarization_model,
        summary_options,
    ):
        captured["checkpointer"] = checkpointer
        captured["model_runtime"] = model_runtime
        captured["store"] = store
        captured["summarization_model"] = summarization_model
        captured["summary_options"] = summary_options
        return FakeRunnable()

    monkeypatch.setattr(runtime_module, "AgentModelRuntime", FakeAgentModelRuntime)
    monkeypatch.setattr(runtime_module, "build_graph", fake_build_graph)

    model_options = ChatModelOptions()
    result = await runtime_module.run_graph(
        checkpointer,
        thread_id="thread-1",
        user_id="user-1",
        message="今天吃什么？",
        model_options=model_options,
        session_factory=FakeSessionFactory(),
        settings=Settings(
            summary_max_tokens=1200,
            summary_trigger_tokens=900,
            summary_max_output_tokens=300,
        ),
        memory_store=memory_store,
    )

    assert result["messages"][-1].content == "ok"
    assert captured["checkpointer"] is checkpointer
    assert checkpointer.config == {
        "configurable": {
            "thread_id": "thread-1",
            "langgraph_user_id": "user-1",
        }
    }
    assert captured["store"] is memory_store
    assert captured["model_options"] is model_options
    assert captured["summary_model_config"].model_name == "test-model"
    assert captured["summarization_model"] is summarization_model
    assert captured["summary_options"].max_tokens == 1200
    assert captured["summary_options"].trigger_tokens == 900
    assert captured["summary_options"].max_output_tokens == 300
    assert isinstance(captured["state"]["messages"][0], HumanMessage)
    assert captured["state"]["messages"][0].content == "今天吃什么？"
    assert captured["config"] == {
        "configurable": {
            "thread_id": "thread-1",
            "langgraph_user_id": "user-1",
        }
    }


@pytest.mark.asyncio
async def test_run_graph_restores_messages_from_postgres_when_checkpoint_is_missing(
    monkeypatch,
) -> None:
    """Redis checkpoint 不存在时，runtime 应该从 PostgreSQL conversation 恢复上下文。"""

    captured = {}
    restored_messages = [
        HumanMessage(content="我的好朋友是小亮"),
        AIMessage(content="听起来小亮对你很重要。"),
        HumanMessage(content="我的好朋友是谁"),
    ]
    checkpointer = FakeCheckpointer(checkpoint=None)
    summarization_model = object()

    class FakeAgentModelRuntime:
        """测试用模型运行时，不访问真实数据库或大模型。"""

        def __init__(self, *, config_resolver):
            self.config_resolver = config_resolver

        async def resolve_config(self, options):
            captured["model_options"] = options
            return EffectiveModelConfig(
                source="env_fallback",
                provider="openai_compatible",
                base_url="https://api.example.com/v1",
                model_name="test-model",
                params={},
                credential=ProviderCredential(
                    id=None,
                    provider="openai_compatible",
                    name="env",
                    api_key="test-key",
                ),
                digest="digest",
            )

        def create_chat_model(self, config):
            captured["summary_model_config"] = config
            return summarization_model

    class FakeConversationContextLoader:
        """测试用 PG 上下文恢复器。"""

        def __init__(self, repository):
            captured["context_repository"] = repository

        async def load_messages(self, *, thread_id):
            captured["restore_thread_id"] = thread_id
            return restored_messages

    class FakeRunnable:
        async def ainvoke(self, state, config):
            captured["state"] = state
            captured["config"] = config
            return {"messages": [AIMessage(content="小亮。")]}

    def fake_build_graph(
        *,
        checkpointer,
        model_runtime,
        store,
        summarization_model,
        summary_options,
    ):
        captured["checkpointer"] = checkpointer
        captured["model_runtime"] = model_runtime
        captured["store"] = store
        captured["summarization_model"] = summarization_model
        captured["summary_options"] = summary_options
        return FakeRunnable()

    monkeypatch.setattr(runtime_module, "AgentModelRuntime", FakeAgentModelRuntime)
    monkeypatch.setattr(
        runtime_module,
        "ConversationContextLoader",
        FakeConversationContextLoader,
    )
    monkeypatch.setattr(runtime_module, "build_graph", fake_build_graph)

    model_options = ChatModelOptions()
    result = await runtime_module.run_graph(
        checkpointer,
        thread_id="thread-1",
        user_id="user-1",
        message="我的好朋友是谁",
        model_options=model_options,
        session_factory=FakeSessionFactory(),
        settings=Settings(),
        memory_store="memory-store",
    )

    assert result["messages"][-1].content == "小亮。"
    assert captured["restore_thread_id"] == "thread-1"
    assert captured["state"]["messages"] == restored_messages
    assert captured["store"] == "memory-store"
    assert captured["summarization_model"] is summarization_model
    assert captured["config"] == {
        "configurable": {
            "thread_id": "thread-1",
            "langgraph_user_id": "user-1",
        }
    }
