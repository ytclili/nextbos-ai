from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_graph
from app.agent.schemas.intent import IntentDecision, IntentFilter
from app.agent.schemas.sql_plan import SqlCandidate, SqlGenerationPlan
from app.agent.schemas.sql_validation import SqlValidationIssue, SqlValidationResult
from app.integrations.business.wren import (
    WrenContextError,
    WrenContextRequest,
    WrenContextResult,
    WrenModel,
)
from app.llm.models import EffectiveModelConfig, ProviderCredential


class FakeAgentModelRuntime:
    """测试图路由用的模型运行时。"""

    def __init__(
        self,
        decision: IntentDecision,
        sql_plan: SqlGenerationPlan | None = None,
    ) -> None:
        self.decision = decision
        self.sql_plan = sql_plan or _sql_plan()
        self.chat_model = FakeChatModel(decision, self.sql_plan)
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

    def __init__(self, decision: IntentDecision, sql_plan: SqlGenerationPlan) -> None:
        self.decision = decision
        self.sql_plan = sql_plan
        self.calls = []
        self.structured_calls = []

    def with_structured_output(self, schema):
        return FakeStructuredChatModel(self, schema)

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="这是直接回答。")


class FakeStructuredChatModel:
    """结构化输出模型，根据 schema 返回不同节点需要的对象。"""

    def __init__(self, chat_model: FakeChatModel, schema: object) -> None:
        self.chat_model = chat_model
        self.schema = schema

    async def ainvoke(self, messages):
        self.chat_model.structured_calls.append({"schema": self.schema, "messages": messages})
        if self.schema is SqlGenerationPlan:
            return self.chat_model.sql_plan
        return self.chat_model.decision


class FakeWrenContextClient:
    """测试用 WrenAI 上下文客户端，记录 graph 传入的请求。"""

    def __init__(self) -> None:
        self.requests: list[WrenContextRequest] = []
        self.sql_validation_calls = []

    async def query_context(self, request: WrenContextRequest) -> WrenContextResult:
        self.requests.append(request)
        return WrenContextResult(
            models=[WrenModel(name="orders", description="订单模型")],
            raw={"source": "fake-wren"},
        )

    async def validate_sql(
        self,
        *,
        candidate_name: str,
        sql: str,
        repair_attempt: int = 0,
        max_repair_attempts: int = 2,
    ) -> SqlValidationResult:
        self.sql_validation_calls.append(
            {
                "candidate_name": candidate_name,
                "sql": sql,
                "repair_attempt": repair_attempt,
                "max_repair_attempts": max_repair_attempts,
            }
        )
        return SqlValidationResult(
            candidate_name=candidate_name,
            sql=sql,
            status="passed",
            validated_sql=sql,
        )


class FailingWrenContextClient:
    """测试用失败 WrenAI 上下文客户端。"""

    async def query_context(self, request: WrenContextRequest) -> WrenContextResult:
        raise WrenContextError("Wren project 配置无效")

    async def validate_sql(
        self,
        *,
        candidate_name: str,
        sql: str,
        repair_attempt: int = 0,
        max_repair_attempts: int = 2,
    ) -> SqlValidationResult:
        return SqlValidationResult(
            candidate_name=candidate_name,
            sql=sql,
            status="failed",
            issues=[
                SqlValidationIssue(
                    issue_type="unknown",
                    severity="error",
                    message="Wren project 配置无效",
                )
            ],
        )


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
async def test_graph_routes_business_data_to_wren_context() -> None:
    """需要业务数据时，第一阶段先进入 WrenAI 上下文分支。"""

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

    assert result["route_target"] == "wren_context"
    assert result["wren_context_request"] == {
        "question": "分析本月华东区域销售额下降原因",
        "tenant_id": None,
        "user_id": None,
        "metrics": ["销售额"],
        "dimensions": ["区域"],
        "filters": [{"field": "区域", "value": "华东"}],
        "time_range": "本月",
    }
    assert result["wren_context"] == {
        "models": [],
        "fields": [],
        "metrics": [],
        "joins": [],
        "business_rules": [],
        "historical_sql": [],
        "raw": {},
    }
    message_texts = [str(message.content) for message in result["messages"]]
    assert any("WrenAI 上下文查询" in text for text in message_texts)
    assert any("分析本月华东区域销售额下降原因" in text for text in message_texts)
    assert "SQL dry-run 校验失败" in result["messages"][-1].content
    assert result["sql_generation_plan"]["preferred_candidate_name"] == "primary"
    assert result["sql_validation_result"]["status"] == "failed"
    assert result["sql_validation_result"]["validated_sql"] is None
    assert model_runtime.chat_model.calls == []


@pytest.mark.asyncio
async def test_graph_business_data_calls_injected_wren_context_client() -> None:
    """配置真实 Wren client 时，业务数据分支应该调用注入的上下文客户端。"""

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
    wren_context_client = FakeWrenContextClient()
    graph = build_graph(
        model_runtime=model_runtime,
        wren_context_client=wren_context_client,
    )

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="分析本月华东销售额下降原因")],
            "model_options": None,
            "user_id": "user-1",
        }
    )

    assert len(wren_context_client.requests) == 1
    assert wren_context_client.requests[0].model_dump() == {
        "question": "分析本月华东区域销售额下降原因",
        "tenant_id": None,
        "user_id": "user-1",
        "metrics": ["销售额"],
        "dimensions": ["区域"],
        "filters": [{"field": "区域", "value": "华东"}],
        "time_range": "本月",
    }
    assert result["wren_context"]["models"] == [
        {"name": "orders", "description": "订单模型"}
    ]
    assert result["wren_context"]["raw"] == {"source": "fake-wren"}
    assert result["sql_generation_plan"]["models"] == ["orders"]
    assert wren_context_client.sql_validation_calls == [
        {
            "candidate_name": "primary",
            "sql": "select order_date, sum(paid_amount) as sales_amount from orders",
            "repair_attempt": 0,
            "max_repair_attempts": 2,
        }
    ]
    assert result["sql_validation_result"]["status"] == "passed"
    assert result["sql_validation_result"]["validated_sql"].startswith("select order_date")
    assert model_runtime.chat_model.calls == []


@pytest.mark.asyncio
async def test_graph_business_data_keeps_state_when_wren_context_fails() -> None:
    """Wren 查询失败时，图不应直接崩溃，而应写入可调试的错误快照。"""

    model_runtime = FakeAgentModelRuntime(
        _decision(
            intent_type="business_analysis",
            needs_business_data=True,
            question_rewrite="分析本月华东区域销售额下降原因",
        )
    )
    graph = build_graph(
        model_runtime=model_runtime,
        wren_context_client=FailingWrenContextClient(),
    )

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="分析本月华东销售额下降原因")],
            "model_options": None,
        }
    )

    assert result["route_target"] == "wren_context"
    assert result["wren_context"]["raw"] == {"error": "Wren project 配置无效"}
    message_texts = [str(message.content) for message in result["messages"]]
    assert any("WrenAI 上下文查询失败" in text for text in message_texts)
    assert "SQL dry-run 校验失败" in result["messages"][-1].content
    assert result["sql_generation_plan"]["status"] == "ready_for_dry_run"
    assert result["sql_validation_result"]["status"] == "failed"
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


def _sql_plan() -> SqlGenerationPlan:
    return SqlGenerationPlan(
        status="ready_for_dry_run",
        query_intent="trend",
        question="分析本月华东区域销售额下降原因",
        planning_summary="按日期聚合订单销售额观察趋势。",
        models=["orders"],
        metrics=["销售额"],
        dimensions=["日期"],
        candidates=[
            SqlCandidate(
                name="primary",
                sql="select order_date, sum(paid_amount) as sales_amount from orders",
                rationale="按日期聚合能观察销售额变化。",
                expected_columns=["order_date", "sales_amount"],
            )
        ],
        preferred_candidate_name="primary",
        confidence=0.8,
    )
