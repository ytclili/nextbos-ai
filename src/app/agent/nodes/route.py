from app.agent.intent import RouteTarget
from app.agent.state import AgentState


def route_after_intent(state: AgentState) -> RouteTarget:
    """读取 intent 节点写入的确定性路由结果。"""

    route_target = state.get("route_target")
    if route_target in {"direct_answer", "wren_context_stub", "clarify"}:
        return route_target
    return "direct_answer"
