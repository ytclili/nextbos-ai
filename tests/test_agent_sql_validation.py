import pytest
from pydantic import ValidationError

from app.agent.schemas.sql_validation import SqlValidationIssue, SqlValidationResult


def test_sql_validation_result_passed_requires_validated_sql() -> None:
    """校验通过时必须显式给出可进入受控执行的 SQL。"""

    result = SqlValidationResult(
        candidate_name="primary",
        sql="select sum(paid_amount) as sales_amount from orders",
        status="passed",
        validated_sql="select sum(paid_amount) as sales_amount from orders limit 1000",
        issues=[
            SqlValidationIssue(
                issue_type="resource",
                severity="warning",
                message="已确认结果行数受 limit 控制。",
            )
        ],
        dry_plan={"ok": True},
        dry_run={"ok": True},
    )

    assert result.validated_sql.endswith("limit 1000")
    assert result.model_dump()["issues"][0]["severity"] == "warning"


def test_sql_validation_result_rejects_passed_without_validated_sql() -> None:
    """没有 validated_sql 时不能标记为 passed。"""

    with pytest.raises(ValidationError, match="必须包含 validated_sql"):
        SqlValidationResult(
            candidate_name="primary",
            sql="select sum(paid_amount) from orders",
            status="passed",
        )


def test_sql_validation_result_rejects_execution_sql_when_not_passed() -> None:
    """未通过校验时不能携带 validated_sql，防止后续节点误执行。"""

    with pytest.raises(ValidationError, match="validated_sql 必须为空"):
        SqlValidationResult(
            candidate_name="primary",
            sql="select * from missing_table",
            status="needs_repair",
            validated_sql="select * from missing_table",
            issues=[
                SqlValidationIssue(
                    issue_type="model",
                    severity="error",
                    message="模型 missing_table 不存在。",
                )
            ],
        )


def test_sql_validation_result_requires_issue_when_failed() -> None:
    """失败或待修复状态必须带紧凑错误，供 SQL 修复节点使用。"""

    with pytest.raises(ValidationError, match="至少一个 issue"):
        SqlValidationResult(
            candidate_name="primary",
            sql="select * from missing_table",
            status="failed",
        )


def test_sql_validation_result_rejects_repair_after_max_attempts() -> None:
    """达到最大修复次数后，状态应该进入 failed，而不是继续 needs_repair。"""

    with pytest.raises(ValidationError, match="最大修复次数"):
        SqlValidationResult(
            candidate_name="primary",
            sql="select * from missing_table",
            status="needs_repair",
            issues=[
                SqlValidationIssue(
                    issue_type="field",
                    severity="error",
                    message="字段 paid_amount 不存在。",
                )
            ],
            repair_attempt=2,
            max_repair_attempts=2,
        )
