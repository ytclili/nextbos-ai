from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.nodes.chart_plan import create_chart_plan_node
from app.agent.schemas.chart import ChartEncoding, ChartPlan
from app.llm.models import EffectiveModelConfig, ProviderCredential


@pytest.mark.asyncio
async def test_chart_plan_node_asks_llm_for_structured_chart_plan() -> None:
    """chart_plan 节点应该让 LLM 选择图表类型和字段映射。"""

    plan = ChartPlan(
        chart_type="line",
        title="华东销售额趋势",
        x=ChartEncoding(field="order_date", label="日期"),
        y=ChartEncoding(field="sales_amount", label="销售额"),
        series=None,
        description="按日期展示华东销售额变化。",
    )
    model_runtime = FakeAgentModelRuntime(plan)
    node = create_chart_plan_node(model_runtime=model_runtime)

    result = await node(
        {
            "messages": [HumanMessage(content="画一下本月华东销售额趋势图")],
            "model_options": None,
            "intent_decision": {"output_type": "chart"},
            "sql_execution_result": {
                "status": "succeeded",
                "columns": [{"name": "order_date"}, {"name": "sales_amount"}],
                "rows": [{"order_date": "2026-09-01", "sales_amount": "100.00"}],
                "row_count": 1,
            },
        }
    )

    assert result["chart_plan_result"] == plan.model_dump()
    assert result["llm_snapshot_id"] == model_runtime.snapshot_id
    assert model_runtime.chat_model.schema is ChartPlan
    assert any(
        isinstance(message, SystemMessage) and "sql_execution_result" in message.content
        for message in model_runtime.chat_model.messages
    )


class FakeAgentModelRuntime:
    """测试用模型运行时，不访问真实模型。"""

    def __init__(self, plan: ChartPlan) -> None:
        self.plan = plan
        self.chat_model = FakeChatModel(plan)
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
    """测试用 ChatModel，只记录 structured output 调用。"""

    def __init__(self, plan: ChartPlan) -> None:
        self.plan = plan
        self.schema = None
        self.messages = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return self.plan
