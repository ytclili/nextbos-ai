import json

import pytest

from app.core.config import Settings
from app.llm.models import EffectiveModelConfig, ProviderCredential


def test_langgraph_config_points_to_studio_graph() -> None:
    """LangGraph Studio 应该使用专门的 Studio 运行入口。"""

    config = json.loads(open("langgraph.json", encoding="utf-8").read())

    assert config["graphs"]["main_agent"] == "app.agent.studio_graph:graph"


def test_create_studio_graph_builds_graph_with_studio_runtime(monkeypatch) -> None:
    """Studio graph 应该注入真实模型运行时，而不是使用 fallback 回显节点。"""

    from app.agent import studio_graph

    captured = {}

    def fake_build_graph(*, model_runtime):
        captured["model_runtime"] = model_runtime
        return "compiled-graph"

    monkeypatch.setattr(studio_graph, "build_graph", fake_build_graph)

    graph = studio_graph.create_studio_graph()

    assert graph == "compiled-graph"
    assert isinstance(captured["model_runtime"], studio_graph.StudioModelRuntime)


@pytest.mark.asyncio
async def test_studio_model_runtime_initializes_schema_once_and_calls_model_runtime(
    monkeypatch,
) -> None:
    """Studio runtime 应该先准备 schema，再用 session 调用 AgentModelRuntime。"""

    from app.agent import studio_graph

    initialized_count = 0
    sessions = []
    runtime_calls = []

    class FakeSessionContext:
        async def __aenter__(self):
            sessions.append("session")
            return "session"

        async def __aexit__(self, exc_type, exc, traceback):
            return None

    class FakeSessionFactory:
        def __call__(self):
            return FakeSessionContext()

    class FakeAgentModelRuntime:
        def __init__(self, *, config_resolver):
            self.config_resolver = config_resolver

        async def resolve_config(self, options):
            runtime_calls.append({"options": options})
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

    async def initialize_schema() -> None:
        nonlocal initialized_count
        initialized_count += 1

    monkeypatch.setattr(studio_graph, "AgentModelRuntime", FakeAgentModelRuntime)

    runtime = studio_graph.StudioModelRuntime(
        settings=Settings(),
        session_factory=FakeSessionFactory(),
        initialize_schema=initialize_schema,
    )

    first = await runtime.resolve_config(options=None)
    second = await runtime.resolve_config(options=None)

    assert first.model_name == "test-model"
    assert second.model_name == "test-model"
    assert initialized_count == 1
    assert sessions == ["session", "session"]
    assert runtime_calls == [
        {"options": None},
        {"options": None},
    ]
