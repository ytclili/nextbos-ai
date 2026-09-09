import pytest
from pydantic import ValidationError

from app.agent.schemas.sql_execution import (
    SqlExecutionColumn,
    SqlExecutionError,
    SqlExecutionResult,
)


def test_sql_execution_result_accepts_successful_dataset() -> None:
    """成功执行时应该保存 SQL、字段和结果行。"""

    result = SqlExecutionResult(
        status="succeeded",
        query_id="query-1",
        sql="select id, total_amount from wms_outbound_order_totals limit 20",
        columns=[
            SqlExecutionColumn(name="id", data_type="BIGINT"),
            SqlExecutionColumn(name="total_amount", data_type="NUMERIC"),
        ],
        rows=[
            {"id": 1, "total_amount": "100.00"},
            {"id": 2, "total_amount": "200.00"},
        ],
        row_count=2,
    )

    assert result.status == "succeeded"
    assert result.row_count == 2
    assert result.error is None


def test_sql_execution_result_rejects_success_without_sql() -> None:
    """成功执行必须记录实际执行 SQL，方便审计和解释。"""

    with pytest.raises(ValidationError, match="succeeded 状态必须包含实际执行 SQL"):
        SqlExecutionResult(status="succeeded")


def test_sql_execution_result_rejects_mismatched_row_count() -> None:
    """row_count 必须和 rows 长度一致，避免后续节点误读数据规模。"""

    with pytest.raises(ValidationError, match="row_count 必须等于 rows 长度"):
        SqlExecutionResult(
            status="succeeded",
            sql="select id from wms_items",
            rows=[{"id": 1}],
            row_count=2,
        )


def test_sql_execution_result_accepts_failed_result_with_error() -> None:
    """失败结果应该只包含紧凑错误，不携带半截数据。"""

    result = SqlExecutionResult(
        status="failed",
        error=SqlExecutionError(
            error_type="database",
            message="数据库执行失败",
        ),
    )

    assert result.status == "failed"
    assert result.rows == []
    assert result.error is not None


def test_sql_execution_result_rejects_failed_result_without_error() -> None:
    """失败状态必须说明原因。"""

    with pytest.raises(ValidationError, match="failed 状态必须包含 error"):
        SqlExecutionResult(status="failed")


def test_sql_execution_result_rejects_data_when_not_succeeded() -> None:
    """失败或跳过时不能携带结果数据，避免被最终回答当成真实查询结果。"""

    with pytest.raises(ValidationError, match="未成功执行时不能包含结果数据"):
        SqlExecutionResult(
            status="skipped",
            columns=[SqlExecutionColumn(name="id")],
        )
