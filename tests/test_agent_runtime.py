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


@pytest.mark.asyncio
async def test_run_graph_passes_memory_store_and_langgraph_user_id(monkeypatch) -> None:
    """runtime 应该把长期记忆 Store 和用户命名空间配置传给 LangGraph。"""

    captured = {}
    checkpointer = object()
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

    def fake_build_graph(*, checkpointer, model_runtime, store, summarization_model):
        captured["checkpointer"] = checkpointer
        captured["model_runtime"] = model_runtime
        captured["store"] = store
        captured["summarization_model"] = summarization_model
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
        settings=Settings(),
        memory_store=memory_store,
    )

    assert result["messages"][-1].content == "ok"
    assert captured["checkpointer"] is checkpointer
    assert captured["store"] is memory_store
    assert captured["model_options"] is model_options
    assert captured["summary_model_config"].model_name == "test-model"
    assert captured["summarization_model"] is summarization_model
    assert isinstance(captured["state"]["messages"][0], HumanMessage)
    assert captured["state"]["messages"][0].content == "今天吃什么？"
    assert captured["config"] == {
        "configurable": {
            "thread_id": "thread-1",
            "langgraph_user_id": "user-1",
        }
    }
