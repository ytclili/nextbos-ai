from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

SqlPlanStatus = Literal["ready_for_dry_run", "needs_clarification", "cannot_plan"]
SqlQueryIntent = Literal["lookup", "aggregation", "comparison", "trend", "diagnosis"]


class SqlPlanFilter(BaseModel):
    """SQL 计划中的过滤条件，来自意图识别和 WrenAI 业务上下文。"""

    field: str = Field(description="业务字段或模型字段，例如区域、渠道、created_at。")
    operator: str = Field(default="=", description="过滤操作符，例如 =、in、between、>=。")
    value: str = Field(description="过滤值，例如华东、本月、2026-08-01 到 2026-08-31。")


class SqlPlanStep(BaseModel):
    """LLM 生成 SQL 前需要明确的单个规划步骤。"""

    objective: str = Field(description="这一步要解决的查询目标。")
    required_context: list[str] = Field(
        default_factory=list,
        description="这一步依赖的模型、字段、指标、Join 或业务规则。",
    )


class SqlCandidate(BaseModel):
    """一条待交给 Wren dry-plan / dry-run 校验的候选 SQL。"""

    name: str = Field(default="primary", description="候选 SQL 名称，便于校验和修复时引用。")
    sql: str = Field(min_length=1, description="候选 SQL，只表示计划结果，不在本节点执行。")
    rationale: str = Field(description="为什么这条 SQL 能回答当前问题。")
    expected_columns: list[str] = Field(
        default_factory=list,
        description="预期返回字段，后续用于结果解释、表格或图表校验。",
    )
    risk_notes: list[str] = Field(
        default_factory=list,
        description="可能需要 dry-run 重点检查的风险，例如 Join、口径或方言。",
    )


class SqlGenerationPlan(BaseModel):
    """SQL 生成节点的结构化输出。

    这个对象只负责生成可校验的 SQL 计划和候选 SQL；
    真正的方言检查、模型字段检查、权限、Limit 和执行超时都必须放在后续节点。
    """

    status: SqlPlanStatus = Field(description="SQL 计划状态，决定后续是否进入 dry-run。")
    query_intent: SqlQueryIntent = Field(description="本次 SQL 查询的业务目的。")
    question: str = Field(min_length=1, description="当前 SQL 计划要回答的问题。")
    planning_summary: str = Field(description="简要说明 SQL 计划思路。")
    models: list[str] = Field(default_factory=list, description="计划使用的 Wren 模型。")
    metrics: list[str] = Field(default_factory=list, description="计划计算或查询的指标。")
    dimensions: list[str] = Field(default_factory=list, description="计划分组、对比或下钻的维度。")
    filters: list[SqlPlanFilter] = Field(default_factory=list, description="计划使用的过滤条件。")
    time_range: str | None = Field(default=None, description="计划使用的时间范围。")
    assumptions: list[str] = Field(default_factory=list, description="生成 SQL 时做出的业务假设。")
    steps: list[SqlPlanStep] = Field(default_factory=list, description="SQL 生成规划步骤。")
    candidates: list[SqlCandidate] = Field(
        default_factory=list,
        max_length=3,
        description="候选 SQL，通常 1 条即可，复杂场景最多 3 条。",
    )
    preferred_candidate_name: str | None = Field(
        default=None,
        description="优先送去 Wren dry-plan / dry-run 的候选 SQL 名称。",
    )
    clarification_question: str | None = Field(
        default=None,
        description="无法生成可靠 SQL 时，向用户提出的澄清问题。",
    )
    confidence: float = Field(ge=0, le=1, description="模型对 SQL 计划可行性的置信度。")

    @model_validator(mode="after")
    def validate_status_contract(self) -> Self:
        """约束 LLM 结构化输出，避免状态和候选 SQL 自相矛盾。"""

        if self.status == "ready_for_dry_run" and not self.candidates:
            raise ValueError("ready_for_dry_run 状态必须至少包含一条候选 SQL")

        if (
            self.status in {"needs_clarification", "cannot_plan"}
            and not self.clarification_question
        ):
            raise ValueError("无法继续生成 SQL 时必须提供澄清问题")

        if self.preferred_candidate_name is None:
            return self

        candidate_names = {candidate.name for candidate in self.candidates}
        if self.preferred_candidate_name not in candidate_names:
            raise ValueError("preferred_candidate_name 必须来自 candidates")
        return self
