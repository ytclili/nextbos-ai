import pytest
from langchain_core.messages import AIMessage

from app.agent.nodes.sql_validate import create_sql_validate_node
from app.agent.schemas.sql_plan import SqlCandidate, SqlGenerationPlan
from app.agent.schemas.sql_validation import SqlValidationResult


class FakeWrenContextClient:
    """测试用 WrenAI 客户端，只记录 SQL 校验请求。"""

    def __init__(self, result: SqlValidationResult) -> None:
        self.result = result
        self.calls = []

    async def validate_sql(
        self,
        *,
        candidate_name: str,
        sql: str,
        repair_attempt: int = 0,
        max_repair_attempts: int = 2,
    ) -> SqlValidationResult:
        self.calls.append(
            {
                "candidate_name": candidate_name,
                "sql": sql,
                "repair_attempt": repair_attempt,
                "max_repair_attempts": max_repair_attempts,
            }
        )
        return self.result


@pytest.mark.asyncio
async def test_sql_validate_node_validates_preferred_candidate() -> None:
    """SQL 校验节点应该选择 preferred candidate 并调用 Wren validate_sql。"""

    validation_result = SqlValidationResult(
        candidate_name="preferred",
        sql="select sum(paid_amount) as sales_amount from orders",
        status="passed",
        validated_sql="select sum(paid_amount) as sales_amount from orders",
    )
    wren_client = FakeWrenContextClient(validation_result)
    node = create_sql_validate_node(wren_context_client=wren_client)

    result = await node(
        {
            "thread_id": "thread-1",
            "user_id": "user-1",
            "sql_generation_plan": _sql_plan().model_dump(),
        }
    )

    assert wren_client.calls == [
        {
            "candidate_name": "preferred",
            "sql": "select sum(paid_amount) as sales_amount from orders",
            "repair_attempt": 0,
            "max_repair_attempts": 2,
        }
    ]
    assert result["sql_validation_result"]["status"] == "passed"
    assert result["sql_validation_result"]["validated_sql"].startswith("select sum")
    assert isinstance(result["messages"][0], AIMessage)
    assert "SQL dry-run 校验通过" in result["messages"][0].content


@pytest.mark.asyncio
async def test_sql_validate_node_fails_closed_without_wren_client() -> None:
    """没有 Wren 校验客户端时，节点应该 fail-closed，不给出 validated_sql。"""

    node = create_sql_validate_node(wren_context_client=None)

    result = await node({"sql_generation_plan": _sql_plan().model_dump()})

    assert result["sql_validation_result"]["status"] == "failed"
    assert result["sql_validation_result"]["validated_sql"] is None
    assert result["sql_validation_result"]["issues"][0]["message"] == "未配置 WrenAI SQL 校验客户端"
    assert "SQL dry-run 校验失败" in result["messages"][0].content


@pytest.mark.asyncio
async def test_sql_validate_node_fails_closed_when_sql_plan_is_invalid() -> None:
    """SQL 规划快照缺失或格式异常时，节点应该 fail-closed。"""

    node = create_sql_validate_node()

    result = await node({"sql_generation_plan": {"status": "ready_for_dry_run"}})

    assert result["sql_validation_result"]["status"] == "failed"
    assert result["sql_validation_result"]["candidate_name"] == "none"
    assert result["sql_validation_result"]["issues"][0]["message"] == "SQL 规划结果为空或格式不合法"


@pytest.mark.asyncio
async def test_sql_validate_node_fails_closed_when_sql_plan_is_not_ready() -> None:
    """SQL 规划还需要澄清时，节点不应该尝试校验 SQL。"""

    node = create_sql_validate_node()
    plan = SqlGenerationPlan(
        status="needs_clarification",
        query_intent="lookup",
        question="查销售额",
        planning_summary="缺少时间范围。",
        clarification_question="你想查询哪个时间范围的销售额？",
        confidence=0.35,
    )

    result = await node({"sql_generation_plan": plan.model_dump()})

    assert result["sql_validation_result"]["status"] == "failed"
    assert (
        result["sql_validation_result"]["issues"][0]["message"]
        == "你想查询哪个时间范围的销售额？"
    )


def _sql_plan() -> SqlGenerationPlan:
    return SqlGenerationPlan(
        status="ready_for_dry_run",
        query_intent="aggregation",
        question="查询本月销售额",
        planning_summary="聚合订单实付金额。",
        candidates=[
            SqlCandidate(
                name="primary",
                sql="select count(*) as order_count from orders",
                rationale="统计订单数。",
            ),
            SqlCandidate(
                name="preferred",
                sql="select sum(paid_amount) as sales_amount from orders",
                rationale="聚合订单实付金额。",
            ),
        ],
        preferred_candidate_name="preferred",
        confidence=0.8,
    )
