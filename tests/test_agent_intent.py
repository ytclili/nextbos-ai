from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai.lib._pydantic import to_strict_json_schema

from app.agent.intent import IntentDecision, IntentFilter
from app.agent.nodes.intent import create_intent_node, route_intent_decision
from app.llm.models import EffectiveModelConfig, ProviderCredential


class FakeAgentModelRuntime:
    """测试用模型运行时，不访问数据库或真实模型。"""

    def __init__(self, decision: IntentDecision) -> None:
        self.decision = decision
        self.resolve_calls: list[object | None] = []
        self.created_configs: list[EffectiveModelConfig] = []
        self.chat_model = FakeChatModel(decision)
        self.snapshot_id = uuid4()

    async def resolve_config(self, options: object | None) -> EffectiveModelConfig:
        self.resolve_calls.append(options)
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
            snapshot_id=self.snapshot_id,
        )

    def create_chat_model(self, config: EffectiveModelConfig):
        self.created_configs.append(config)
        return self.chat_model


class FakeChatModel:
    """测试用 ChatModel，只记录 structured output 调用。"""

    def __init__(self, decision: IntentDecision) -> None:
        self.decision = decision
        self.schema = None
        self.messages = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return self.decision


def test_route_intent_decision_uses_deterministic_backend_rules() -> None:
    """后端规则应该根据结构化意图稳定选择下一跳。"""

    assert route_intent_decision(_decision(needs_business_data=False)) == "direct_answer"
    assert route_intent_decision(_decision(needs_business_data=True)) == "wren_context_stub"
    assert (
        route_intent_decision(
            _decision(
                intent_type="clarification",
                needs_business_data=True,
                missing_slots=["time_range"],
            )
        )
        == "clarify"
    )


def test_intent_decision_schema_is_compatible_with_openai_structured_output() -> None:
    """OpenAI 严格结构化输出要求所有字段必填，并禁止开放对象字段。"""

    schema = to_strict_json_schema(IntentDecision)

    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    assert schema["properties"]["filters"]["type"] == "array"
    assert (
        schema["$defs"]["IntentFilter"]["required"]
        == list(schema["$defs"]["IntentFilter"]["properties"])
    )
    assert schema["$defs"]["IntentFilter"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_intent_node_passes_context_and_current_question_to_structured_model() -> None:
    """意图节点应该保留上下文，同时标注当前要识别的用户问题。"""

    decision = _decision(
        intent_type="business_analysis",
        needs_business_data=True,
        question_rewrite="分析本月华东区域销售额下降原因",
        metrics=["销售额"],
        dimensions=["区域"],
        filters={"区域": "华东"},
        time_range="本月",
    )
    model_runtime = FakeAgentModelRuntime(decision)
    node = create_intent_node(model_runtime=model_runtime)
    history = [
        HumanMessage(content="分析本月销售额下降原因"),
        AIMessage(content="你想看哪个区域？"),
        HumanMessage(content="华东"),
    ]

    result = await node(
        {
            "messages": history,
            "model_options": None,
            "thread_id": "thread-1",
            "user_id": "user-1",
        }
    )

    assert model_runtime.resolve_calls == [None]
    assert len(model_runtime.created_configs) == 1
    assert model_runtime.chat_model.schema is IntentDecision
    assert isinstance(model_runtime.chat_model.messages[0], SystemMessage)
    assert model_runtime.chat_model.messages[1:4] == history
    assert "当前需要识别意图的用户问题是" in model_runtime.chat_model.messages[-1].content
    assert "华东" in model_runtime.chat_model.messages[-1].content
    assert result["intent_decision"]["question_rewrite"] == "分析本月华东区域销售额下降原因"
    assert result["route_target"] == "wren_context_stub"
    assert result["llm_snapshot_id"] == model_runtime.snapshot_id


def _decision(
    *,
    intent_type: str = "direct_answer",
    needs_business_data: bool,
    output_type: str = "text",
    question_rewrite: str = "解释同比和环比的区别",
    metrics: list[str] | None = None,
    dimensions: list[str] | None = None,
    filters: dict[str, str] | None = None,
    time_range: str | None = None,
    confidence: float = 0.9,
    missing_slots: list[str] | None = None,
    clarification_question: str | None = None,
) -> IntentDecision:
    return IntentDecision(
        intent_type=intent_type,
        needs_business_data=needs_business_data,
        output_type=output_type,
        question_rewrite=question_rewrite,
        metrics=metrics or [],
        dimensions=dimensions or [],
        filters=[
            IntentFilter(field=field, value=value) for field, value in (filters or {}).items()
        ],
        time_range=time_range,
        confidence=confidence,
        missing_slots=missing_slots or [],
        clarification_question=clarification_question,
    )
