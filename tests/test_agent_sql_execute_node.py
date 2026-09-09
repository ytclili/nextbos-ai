import pytest

from app.agent.nodes.sql_execute import create_sql_execute_node
from app.agent.schemas.sql_execution import SqlExecutionResult
from app.agent.schemas.sql_validation import SqlValidationIssue, SqlValidationResult


class FakeSqlExecutionClient:
    """测试用 SQL 执行客户端。"""

    def __init__(self, result: SqlExecutionResult) -> None:
        self.result = result
        self.calls = []

    async def execute_sql(self, *, sql: str, query_id: str | None = None, max_rows: int = 100):
        self.calls.append({"sql": sql, "query_id": query_id, "max_rows": max_rows})
        return self.result


@pytest.mark.asyncio
async def test_sql_execute_node_executes_passed_validated_sql() -> None:
    """SQL 校验通过时，节点应该执行 validated_sql。"""

    execution_result = SqlExecutionResult(
        status="succeeded",
        sql="select id from wms_items limit 100",
        columns=[],
        rows=[],
        row_count=0,
    )
    execution_client = FakeSqlExecutionClient(execution_result)
    node = create_sql_execute_node(sql_execution_client=execution_client, max_rows=50)

    result = await node(
        {
            "sql_validation_result": SqlValidationResult(
                candidate_name="primary",
                sql="select id from wms_items",
                status="passed",
                validated_sql="select id from wms_items limit 100",
            ).model_dump()
        }
    )

    assert execution_client.calls[0]["sql"] == "select id from wms_items limit 100"
    assert execution_client.calls[0]["query_id"]
    assert execution_client.calls[0]["max_rows"] == 50
    assert result["sql_execution_result"]["status"] == "succeeded"
    assert "SQL 执行成功" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_sql_execute_node_marks_empty_or_null_result_for_review() -> None:
    """SQL 执行成功但结果为空或全为空值时，节点应该标记需要复核。"""

    execution_result = SqlExecutionResult(
        status="succeeded",
        sql="select sum(available_inventory) as remaining_quantity_roots from wms_items",
        columns=[{"name": "remaining_quantity_roots"}],
        rows=[{"remaining_quantity_roots": None}],
        row_count=1,
    )
    execution_client = FakeSqlExecutionClient(execution_result)
    node = create_sql_execute_node(sql_execution_client=execution_client)

    result = await node(
        {
            "sql_validation_result": SqlValidationResult(
                candidate_name="primary",
                sql="select sum(available_inventory) from wms_items",
                status="passed",
                validated_sql=(
                    "select sum(available_inventory) as remaining_quantity_roots "
                    "from wms_items"
                ),
            ).model_dump()
        }
    )

    assert result["sql_execution_result"]["status"] == "succeeded"
    assert result["sql_result_review"]["needs_review"] is True
    assert result["sql_result_review"]["reason"] == "SQL 执行成功但结果为空或全为空值"


@pytest.mark.asyncio
async def test_sql_execute_node_skips_when_validation_not_passed() -> None:
    """SQL 未通过校验时，节点不能执行 SQL。"""

    execution_client = FakeSqlExecutionClient(
        SqlExecutionResult(status="succeeded", sql="select 1", row_count=0)
    )
    node = create_sql_execute_node(sql_execution_client=execution_client)

    result = await node(
        {
            "sql_validation_result": SqlValidationResult(
                candidate_name="primary",
                sql="select id from wms_items",
                status="failed",
                issues=[
                    SqlValidationIssue(
                        issue_type="field",
                        severity="error",
                        message="字段不存在",
                    )
                ],
            ).model_dump()
        }
    )

    assert execution_client.calls == []
    assert result["sql_execution_result"]["status"] == "skipped"
    assert "SQL 执行已跳过" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_sql_execute_node_skips_without_execution_client() -> None:
    """没有执行客户端时，节点应该跳过执行。"""

    node = create_sql_execute_node(sql_execution_client=None)

    result = await node(
        {
            "sql_validation_result": SqlValidationResult(
                candidate_name="primary",
                sql="select id from wms_items",
                status="passed",
                validated_sql="select id from wms_items",
            ).model_dump()
        }
    )

    assert result["sql_execution_result"]["status"] == "skipped"
    assert result["sql_execution_result"]["error"]["message"] == "未配置 SQL 执行客户端"


@pytest.mark.asyncio
async def test_sql_execute_node_skips_invalid_validation_snapshot() -> None:
    """SQL 校验快照格式异常时，节点应该跳过执行。"""

    node = create_sql_execute_node(
        sql_execution_client=FakeSqlExecutionClient(
            SqlExecutionResult(status="succeeded", sql="select 1", row_count=0)
        )
    )

    result = await node({"sql_validation_result": {"status": "passed"}})

    assert result["sql_execution_result"]["status"] == "skipped"
    assert result["sql_execution_result"]["error"]["message"] == "SQL 校验结果为空或格式不合法"
