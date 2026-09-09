from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from app.agent.schemas.sql_plan import SqlCandidate, SqlGenerationPlan
from app.agent.schemas.sql_validation import SqlValidationIssue, SqlValidationResult
from app.agent.state import AgentState
from app.integrations.business.wren import WrenContextClient

SqlValidateNode = Callable[[AgentState], Awaitable[dict[str, Any]]]


def create_sql_validate_node(
    *,
    wren_context_client: WrenContextClient | None = None,
) -> SqlValidateNode:
    """创建 SQL 校验节点。

    这个节点只把 SQL 规划节点产出的候选 SQL 送入 Wren dry-plan / dry-run；
    通过校验的 SQL 也只是写入 state，真正数据库执行必须由后续受控执行节点完成。
    """

    async def sql_validate(state: AgentState) -> dict[str, Any]:
        """校验 SQL 候选，并把 SqlValidationResult 写入 LangGraph state。"""

        plan = _sql_generation_plan_from_state(state)
        candidate = _preferred_candidate(plan) if plan else None

        if plan is None:
            result = _failed_validation_result(message="SQL 规划结果为空或格式不合法")
        elif plan.status != "ready_for_dry_run":
            result = _failed_validation_result(
                message=plan.clarification_question or "SQL 规划尚未生成可校验候选"
            )
        elif candidate is None:
            result = _failed_validation_result(message="SQL 规划没有可校验候选")
        elif wren_context_client is None:
            result = _failed_validation_result(
                candidate=candidate,
                message="未配置 WrenAI SQL 校验客户端",
            )
        else:
            result = await wren_context_client.validate_sql(
                candidate_name=candidate.name,
                sql=candidate.sql,
            )

        return {
            "sql_validation_result": result.model_dump(),
            "messages": [AIMessage(content=_validation_message(result))],
        }

    return sql_validate


def _sql_generation_plan_from_state(state: AgentState) -> SqlGenerationPlan | None:
    """从 state 读取 SQL 规划快照；格式异常时返回 None，避免图直接崩溃。"""

    try:
        return SqlGenerationPlan.model_validate(state.get("sql_generation_plan") or {})
    except ValidationError:
        return None


def _preferred_candidate(plan: SqlGenerationPlan) -> SqlCandidate | None:
    """优先选择 preferred_candidate_name，否则选择第一条候选 SQL。"""

    if not plan.candidates:
        return None
    if plan.preferred_candidate_name is None:
        return plan.candidates[0]
    for candidate in plan.candidates:
        if candidate.name == plan.preferred_candidate_name:
            return candidate
    return None


def _failed_validation_result(
    *,
    message: str,
    candidate: SqlCandidate | None = None,
) -> SqlValidationResult:
    """构造 fail-closed 的 SQL 校验失败快照。"""

    return SqlValidationResult(
        candidate_name=candidate.name if candidate else "none",
        sql=candidate.sql if candidate else "/* no sql candidate */",
        status="failed",
        issues=[
            SqlValidationIssue(
                issue_type="unknown",
                severity="error",
                message=message,
            )
        ],
    )


def _validation_message(result: SqlValidationResult) -> str:
    """把 SQL 校验状态转换成 Studio 可读消息。"""

    if result.status == "passed":
        return f"SQL dry-run 校验通过，候选：{result.candidate_name}"
    if result.status == "needs_repair":
        return f"SQL dry-run 需要修复，候选：{result.candidate_name}"
    return f"SQL dry-run 校验失败，候选：{result.candidate_name}"
