from contextlib import contextmanager

import pytest

from app.agent.model_runtime import AgentModelRuntime
from app.agent.options import ChatModelOptions
from app.llm.models import EffectiveModelConfig, ProviderCredential


class RecordingSpan:
    """记录测试里写入的 span attribute。"""

    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class RecordingTracer:
    """测试用 tracer，避免依赖真实 OTel SDK 导出。"""

    def __init__(self):
        self.spans = []

    @contextmanager
    def start_as_current_span(self, _name):
        span = RecordingSpan()
        self.spans.append(span)
        yield span


class FakeConfigResolver:
    """记录模型选择参数，并返回固定的最终模型配置。"""

    def __init__(self):
        self.selection = None

    async def resolve(self, *, selection, requested_params):
        self.selection = selection
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


@pytest.mark.asyncio
async def test_resolve_config_records_empty_model_alias_when_option_alias_is_none(monkeypatch):
    from app.agent import model_runtime

    tracer = RecordingTracer()
    monkeypatch.setattr(model_runtime, "tracer", tracer)
    resolver = FakeConfigResolver()
    runtime = AgentModelRuntime(config_resolver=resolver)

    await runtime.resolve_config(ChatModelOptions(model_alias=None))

    assert resolver.selection.model_alias is None
    assert tracer.spans[0].attributes["llm.request.model_alias"] == ""
