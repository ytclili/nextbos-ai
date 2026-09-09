import pytest
from pydantic import ValidationError

from app.agent.schemas.sql_plan import (
    SqlCandidate,
    SqlGenerationPlan,
    SqlPlanFilter,
    SqlPlanStep,
)


def test_sql_generation_plan_ready_for_dry_run_serializes_candidates() -> None:
    """可 dry-run 的 SQL 计划应该稳定序列化，供 LangGraph state 保存。"""

    plan = SqlGenerationPlan(
        status="ready_for_dry_run",
        query_intent="diagnosis",
        question="分析本月华东销售额下降原因",
        planning_summary="按区域过滤华东，并按日期聚合销售额观察趋势。",
        models=["orders"],
        metrics=["销售额"],
        dimensions=["日期"],
        filters=[SqlPlanFilter(field="region", value="华东")],
        time_range="本月",
        assumptions=["销售额使用订单实付金额口径。"],
        steps=[
            SqlPlanStep(
                objective="查询华东本月每日销售额",
                required_context=["orders.region", "orders.paid_amount", "orders.created_at"],
            )
        ],
        candidates=[
            SqlCandidate(
                name="daily_sales",
                sql=(
                    "select created_date, sum(paid_amount) as sales_amount "
                    "from orders group by created_date"
                ),
                rationale="每日聚合能观察销售额下降出现在哪些日期。",
                expected_columns=["created_date", "sales_amount"],
            )
        ],
        preferred_candidate_name="daily_sales",
        confidence=0.86,
    )

    assert plan.model_dump()["candidates"][0]["name"] == "daily_sales"
    assert plan.model_dump()["filters"] == [
        {"field": "region", "operator": "=", "value": "华东"}
    ]


def test_sql_generation_plan_requires_candidate_when_ready() -> None:
    """ready_for_dry_run 必须带候选 SQL，不能让后续 dry-run 无输入。"""

    with pytest.raises(ValidationError, match="至少包含一条候选 SQL"):
        SqlGenerationPlan(
            status="ready_for_dry_run",
            query_intent="aggregation",
            question="查询本月销售额",
            planning_summary="按本月过滤后聚合销售额。",
            confidence=0.7,
        )


def test_sql_generation_plan_requires_clarification_question_when_blocked() -> None:
    """需要澄清或无法规划时，必须给出用户可回答的问题。"""

    with pytest.raises(ValidationError, match="必须提供澄清问题"):
        SqlGenerationPlan(
            status="needs_clarification",
            query_intent="lookup",
            question="查销售额",
            planning_summary="缺少时间范围。",
            confidence=0.4,
        )


def test_sql_generation_plan_preferred_candidate_must_exist() -> None:
    """优先候选名称必须来自 candidates，避免后续节点引用不存在的 SQL。"""

    with pytest.raises(ValidationError, match="必须来自 candidates"):
        SqlGenerationPlan(
            status="ready_for_dry_run",
            query_intent="aggregation",
            question="查询本月销售额",
            planning_summary="按本月过滤后聚合销售额。",
            candidates=[
                SqlCandidate(
                    name="primary",
                    sql="select sum(paid_amount) as sales_amount from orders",
                    rationale="聚合订单实付金额。",
                )
            ],
            preferred_candidate_name="missing",
            confidence=0.7,
        )
