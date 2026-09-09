from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.nodes.sql_plan import create_sql_plan_node
from app.agent.schemas.sql_plan import SqlCandidate, SqlGenerationPlan
from app.agent.state import AgentState
from app.llm.models import EffectiveModelConfig, ProviderCredential


class FakeAgentModelRuntime:
    """测试用模型运行时，不访问数据库或真实模型。"""

    def __init__(self, plan: SqlGenerationPlan) -> None:
        self.plan = plan
        self.resolve_calls: list[object | None] = []
        self.created_configs: list[EffectiveModelConfig] = []
        self.chat_model = FakeChatModel(plan)
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

    def __init__(self, plan: SqlGenerationPlan) -> None:
        self.plan = plan
        self.schema = None
        self.messages = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return self.plan


class FakeInvalidPlanRuntime(FakeAgentModelRuntime):
    """测试用模型运行时，模拟 LLM 返回了无法通过 SqlGenerationPlan 校验的内容。"""

    def __init__(self) -> None:
        super().__init__(_ready_sql_plan())
        self.chat_model = FakeInvalidPlanChatModel(self.plan)


class FakeInvalidPlanChatModel(FakeChatModel):
    """测试用 ChatModel，模拟 OpenAI structured output 解析阶段抛出 ValidationError。"""

    async def ainvoke(self, messages):
        self.messages = messages
        return SqlGenerationPlan(
            status="cannot_plan",
            query_intent="lookup",
            question="查销售额",
            planning_summary="缺少可用业务上下文。",
            confidence=0.2,
        )


def test_agent_state_declares_sql_generation_plan_snapshot() -> None:
    """AgentState 应该声明 SQL 规划快照，便于 Studio 展示和 checkpoint。"""

    assert "sql_generation_plan" in AgentState.__annotations__


def test_agent_state_declares_sql_validation_result_snapshot() -> None:
    """AgentState 应该声明 SQL 校验快照，避免后续节点误执行未通过 SQL。"""

    assert "sql_validation_result" in AgentState.__annotations__


@pytest.mark.asyncio
async def test_sql_plan_node_passes_intent_and_wren_context_to_structured_model() -> None:
    """SQL 规划节点应该把意图和 Wren 上下文交给结构化模型。"""

    plan = SqlGenerationPlan(
        status="ready_for_dry_run",
        query_intent="trend",
        question="分析本月华东区域销售额下降原因",
        planning_summary="按日期聚合华东销售额。",
        models=["orders"],
        metrics=["销售额"],
        dimensions=["日期"],
        candidates=[
            SqlCandidate(
                name="daily_sales",
                sql="select order_date, sum(paid_amount) as sales_amount from orders",
                rationale="按日期观察销售额趋势。",
                expected_columns=["order_date", "sales_amount"],
            )
        ],
        preferred_candidate_name="daily_sales",
        confidence=0.82,
    )
    model_runtime = FakeAgentModelRuntime(plan)
    node = create_sql_plan_node(model_runtime=model_runtime)
    history = [HumanMessage(content="分析本月华东销售额下降原因")]

    result = await node(
        {
            "messages": history,
            "model_options": None,
            "thread_id": "thread-1",
            "user_id": "user-1",
            "intent_decision": {
                "question_rewrite": "分析本月华东区域销售额下降原因",
                "metrics": ["销售额"],
                "dimensions": ["区域"],
                "time_range": "本月",
            },
            "wren_context_request": {
                "question": "分析本月华东区域销售额下降原因",
                "metrics": ["销售额"],
            },
            "wren_context": {
                "models": [{"name": "orders", "description": "订单模型"}],
                "fields": [{"model": "orders", "name": "paid_amount"}],
            },
        }
    )

    assert model_runtime.resolve_calls == [None]
    assert len(model_runtime.created_configs) == 1
    assert model_runtime.chat_model.schema is SqlGenerationPlan
    assert isinstance(model_runtime.chat_model.messages[0], SystemMessage)
    assert "SQL 规划节点" in model_runtime.chat_model.messages[0].content
    assert model_runtime.chat_model.messages[1:2] == history
    assert "当前需要生成 SQL 计划的结构化输入是" in model_runtime.chat_model.messages[-1].content
    assert "分析本月华东区域销售额下降原因" in model_runtime.chat_model.messages[-1].content
    assert "orders" in model_runtime.chat_model.messages[-1].content
    assert result["sql_generation_plan"]["preferred_candidate_name"] == "daily_sales"
    assert result["llm_snapshot_id"] == model_runtime.snapshot_id


@pytest.mark.asyncio
async def test_sql_plan_node_uses_wren_request_question_when_intent_rewrite_is_missing() -> None:
    """缺少 intent 改写时，SQL 规划节点应该使用 Wren 请求里的问题。"""

    plan = SqlGenerationPlan(
        status="needs_clarification",
        query_intent="lookup",
        question="查销售额",
        planning_summary="缺少时间范围。",
        clarification_question="你想查询哪个时间范围的销售额？",
        confidence=0.35,
    )
    model_runtime = FakeAgentModelRuntime(plan)
    node = create_sql_plan_node(model_runtime=model_runtime)

    await node(
        {
            "messages": [HumanMessage(content="查销售额")],
            "model_options": None,
            "wren_context_request": {"question": "查询销售额"},
        }
    )

    assert '"current_question": "查询销售额"' in model_runtime.chat_model.messages[-1].content


@pytest.mark.asyncio
async def test_sql_plan_node_falls_back_when_structured_output_validation_fails() -> None:
    """LLM 少填必填业务约束时，SQL 规划节点不应该让 Studio run 崩溃。"""

    model_runtime = FakeInvalidPlanRuntime()
    node = create_sql_plan_node(model_runtime=model_runtime)

    result = await node(
        {
            "messages": [HumanMessage(content="查销售额")],
            "model_options": None,
        }
    )

    assert result["sql_generation_plan"]["status"] == "cannot_plan"
    assert result["sql_generation_plan"]["candidates"] == []
    assert result["sql_generation_plan"]["clarification_question"]


def _ready_sql_plan() -> SqlGenerationPlan:
    return SqlGenerationPlan(
        status="ready_for_dry_run",
        query_intent="aggregation",
        question="查询本月销售额",
        planning_summary="聚合订单实付金额。",
        candidates=[
            SqlCandidate(
                name="primary",
                sql="select sum(paid_amount) as sales_amount from orders",
                rationale="聚合订单实付金额。",
            )
        ],
        preferred_candidate_name="primary",
        confidence=0.8,
    )
