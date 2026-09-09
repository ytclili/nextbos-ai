from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

SqlExecutionStatus = Literal["succeeded", "failed", "skipped"]


class SqlExecutionColumn(BaseModel):
    """SQL 执行结果中的单个字段描述。"""

    name: str = Field(min_length=1, description="结果字段名。")
    data_type: str | None = Field(default=None, description="结果字段类型，取不到时为空。")


class SqlExecutionError(BaseModel):
    """受控 SQL 执行失败时返回的紧凑错误。"""

    message: str = Field(min_length=1, description="可展示或给后续修复节点使用的错误信息。")
    error_type: str = Field(
        default="unknown",
        description="错误类型，例如 timeout、database、policy。",
    )


class SqlExecutionResult(BaseModel):
    """受控 SQL 执行结果快照。

    这个对象只保存后续回答、表格和图表节点需要的数据摘要；
    SQL 执行节点必须只执行已经通过校验的 validated_sql。
    """

    status: SqlExecutionStatus = Field(description="执行状态。")
    query_id: str | None = Field(default=None, description="本次查询 ID，用于排查和证据引用。")
    sql: str | None = Field(default=None, description="实际提交给受控执行层的 SQL。")
    columns: list[SqlExecutionColumn] = Field(default_factory=list, description="结果字段列表。")
    rows: list[dict[str, Any]] = Field(default_factory=list, description="结果数据行。")
    row_count: int = Field(default=0, ge=0, description="返回给 agent 的行数。")
    truncated: bool = Field(default=False, description="结果是否因为 Limit 或大小限制被截断。")
    error: SqlExecutionError | None = Field(default=None, description="执行失败原因。")

    @model_validator(mode="after")
    def validate_status_contract(self) -> Self:
        """约束执行状态，避免失败结果被后续节点当成真实数据使用。"""

        if self.status == "succeeded":
            if not self.sql:
                raise ValueError("succeeded 状态必须包含实际执行 SQL")
            if self.error is not None:
                raise ValueError("succeeded 状态不能包含 error")
            if self.row_count != len(self.rows):
                raise ValueError("row_count 必须等于 rows 长度")
            return self

        if self.rows or self.columns:
            raise ValueError("未成功执行时不能包含结果数据")

        if self.status == "failed" and self.error is None:
            raise ValueError("failed 状态必须包含 error")

        return self
