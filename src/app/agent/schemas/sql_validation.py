from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

SqlValidationStatus = Literal["passed", "needs_repair", "failed"]
SqlValidationIssueType = Literal[
    "model",
    "field",
    "join",
    "dialect",
    "syntax",
    "permission",
    "resource",
    "unknown",
]
SqlValidationIssueSeverity = Literal["warning", "error"]


class SqlValidationIssue(BaseModel):
    """Wren dry-plan / dry-run 返回的单个 SQL 校验问题。"""

    issue_type: SqlValidationIssueType = Field(description="问题类型，例如字段、Join 或方言错误。")
    severity: SqlValidationIssueSeverity = Field(description="问题严重程度。")
    message: str = Field(min_length=1, description="可给 LLM 修复 SQL 使用的紧凑错误信息。")
    detail: str | None = Field(default=None, description="可选的底层错误详情，不能包含敏感数据。")
    suggested_fix: str | None = Field(default=None, description="可选修复建议。")


class SqlValidationResult(BaseModel):
    """候选 SQL 的 Wren dry-plan / dry-run 校验结果。

    这个对象只表示 SQL 是否能进入后续受控执行；
    即使 status=passed，也不代表 SQL 已经执行或已经返回业务数据。
    """

    candidate_name: str = Field(min_length=1, description="被校验的候选 SQL 名称。")
    sql: str = Field(min_length=1, description="被校验的候选 SQL。")
    status: SqlValidationStatus = Field(description="校验状态，决定后续执行、修复或失败。")
    validated_sql: str | None = Field(
        default=None,
        description="通过校验、可交给受控执行节点的 SQL；未通过时必须为空。",
    )
    issues: list[SqlValidationIssue] = Field(
        default_factory=list,
        description="Wren 返回的模型、字段、Join、语法、权限或资源风险问题。",
    )
    dry_plan: dict[str, Any] = Field(default_factory=dict, description="Wren dry-plan 原始摘要。")
    dry_run: dict[str, Any] = Field(default_factory=dict, description="Wren dry-run 原始摘要。")
    repair_attempt: int = Field(default=0, ge=0, description="当前 SQL 修复尝试次数。")
    max_repair_attempts: int = Field(default=2, ge=0, le=5, description="最多允许修复次数。")

    @model_validator(mode="after")
    def validate_status_contract(self) -> Self:
        """约束校验状态，避免后续节点误执行未通过 SQL。"""

        if self.status == "passed":
            if not self.validated_sql:
                raise ValueError("passed 状态必须包含 validated_sql")
            blocking_errors = [issue for issue in self.issues if issue.severity == "error"]
            if blocking_errors:
                raise ValueError("passed 状态不能包含 error 级别问题")
            return self

        if self.validated_sql is not None:
            raise ValueError("未通过校验时 validated_sql 必须为空")

        if self.status in {"needs_repair", "failed"} and not self.issues:
            raise ValueError("未通过校验时必须包含至少一个 issue")

        if self.status == "needs_repair" and self.repair_attempt >= self.max_repair_attempts:
            raise ValueError("达到最大修复次数后不能继续标记为 needs_repair")

        return self
