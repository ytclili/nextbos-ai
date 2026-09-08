from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_graph
from app.agent.intent import IntentDecision, IntentFilter
from app.llm.models import EffectiveModelConfig, ProviderCredential


class FakeAgentModelRuntime:
    """测试图路由用的模型运行时。"""

    def __init__(self, decision: IntentDecision) -> None:
        self.decision = decision
        self.chat_model = FakeChatModel(decision)
        self.snapshot_id = uuid4()

    async def resolve_config(self, options: object | None) -> EffectiveModelConfig:
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
        return self.chat_model


class FakeChatModel:
    """同时模拟 intent 结构化输出和 direct answer 普通回复。"""

    def __init__(self, decision: IntentDecision) -> None:
        self.decision = decision
        self.calls = []
        self.structured_calls = []

    def with_structured_output(self, _schema):
        return FakeStructuredChatModel(self)

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="这是直接回答。")


class FakeStructuredChatModel:
    """intent 节点使用的结构化输出模型。"""

    def __init__(self, chat_model: FakeChatModel) -> None:
        self.chat_model = chat_model

    async def ainvoke(self, messages):
        self.chat_model.structured_calls.append(messages)
        return self.chat_model.decision


@pytest.mark.asyncio
async def test_graph_routes_direct_answer_to_respond() -> None:
    """不需要业务数据时，图应该进入直接回答分支。"""

    model_runtime = FakeAgentModelRuntime(_decision(needs_business_data=False))
    graph = build_graph(model_runtime=model_runtime)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="解释一下同比和环比的区别")],
            "model_options": None,
        }
    )

    assert result["route_target"] == "direct_answer"
    assert result["messages"][-1].content == "这是直接回答。"
    assert len(model_runtime.chat_model.calls) == 1


@pytest.mark.asyncio
async def test_graph_routes_business_data_to_wren_context_stub() -> None:
    """需要业务数据时，第一阶段先进入 WrenAI 上下文占位分支。"""

    model_runtime = FakeAgentModelRuntime(
        _decision(
            intent_type="business_analysis",
            needs_business_data=True,
            question_rewrite="分析本月华东区域销售额下降原因",
            metrics=["销售额"],
            dimensions=["区域"],
            filters={"区域": "华东"},
            time_range="本月",
        )
    )
    graph = build_graph(model_runtime=model_runtime)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="分析本月华东销售额下降原因")],
            "model_options": None,
        }
    )

    assert result["route_target"] == "wren_context_stub"
    assert "WrenAI 上下文查询" in result["messages"][-1].content
    assert model_runtime.chat_model.calls == []


@pytest.mark.asyncio
async def test_graph_routes_missing_slots_to_clarify() -> None:
    """缺少必要查询条件时，图应该进入澄清分支。"""

    clarification_question = "你想分析哪个时间范围的销售额下降原因？"
    model_runtime = FakeAgentModelRuntime(
        _decision(
            intent_type="clarification",
            needs_business_data=True,
            question_rewrite="分析销售额下降原因",
            metrics=["销售额"],
            missing_slots=["time_range"],
            clarification_question=clarification_question,
        )
    )
    graph = build_graph(model_runtime=model_runtime)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="分析销售额下降原因")],
            "model_options": None,
        }
    )

    assert result["route_target"] == "clarify"
    assert result["messages"][-1].content == clarification_question
    assert model_runtime.chat_model.calls == []


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
