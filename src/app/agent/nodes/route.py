from typing import Literal

from app.agent.schemas.intent import RouteTarget
from app.agent.state import AgentState

SqlPlanRouteTarget = Literal["sql_validate", "clarify"]


def route_after_intent(state: AgentState) -> RouteTarget:
    """读取 intent 节点写入的确定性路由结果。"""

    route_target = state.get("route_target")
    if route_target in {"direct_answer", "wren_context", "clarify"}:
        return route_target
    return "direct_answer"


def route_after_sql_plan(state: AgentState) -> SqlPlanRouteTarget:
    """SQL 计划没准备好时直接澄清，不进入 dry-plan / dry-run 校验。"""

    plan = state.get("sql_generation_plan") or {}
    if plan.get("status") == "ready_for_dry_run" and plan.get("candidates"):
        return "sql_validate"
    return "clarify"
