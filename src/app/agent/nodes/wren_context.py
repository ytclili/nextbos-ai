from langchain_core.messages import AIMessage

from app.agent.state import AgentState
from app.integrations.business.wren import (
    WrenContextClient,
    WrenContextError,
    WrenContextFilter,
    WrenContextRequest,
    WrenContextResult,
)


def create_wren_context_node(
    *,
    wren_context_client: WrenContextClient | None = None,
):
    """创建 WrenAI 上下文节点。

    没有配置真实客户端时，仍写入稳定的空上下文快照，方便 Studio 调试图结构。
    """

    async def node(state: AgentState) -> dict[str, object]:
        """查询 WrenAI 上下文，并把请求和结果都写入 state。"""

        request = _wren_context_request_from_state(state)
        context = WrenContextResult()
        message = f"已识别为业务数据问题，已准备 WrenAI 上下文查询请求：{request.question}"

        if wren_context_client is not None:
            try:
                context = await wren_context_client.query_context(request)
                message = f"已识别为业务数据问题，已完成 WrenAI 上下文查询：{request.question}"
            except WrenContextError as exc:
                context = WrenContextResult(raw={"error": str(exc)})
                message = f"已识别为业务数据问题，但 WrenAI 上下文查询失败：{request.question}"

        return {
            "wren_context_request": request.model_dump(),
            "wren_context": context.model_dump(),
            "messages": [AIMessage(content=message)],
        }

    return node


wren_context = create_wren_context_node()


def _wren_context_request_from_state(state: AgentState) -> WrenContextRequest:
    """从 AgentState 中提取 WrenAI 上下文查询请求。"""

    decision = state.get("intent_decision") or {}
    filters = [
        WrenContextFilter(field=filter_item["field"], value=filter_item["value"])
        for filter_item in decision.get("filters", [])
    ]
    return WrenContextRequest(
        question=decision.get("question_rewrite") or _latest_message_content(state),
        user_id=state.get("user_id"),
        metrics=decision.get("metrics", []),
        dimensions=decision.get("dimensions", []),
        filters=filters,
        time_range=decision.get("time_range"),
    )


def _latest_message_content(state: AgentState) -> str:
    """没有意图改写时，兜底读取最后一条消息内容。"""

    messages = state.get("messages", [])
    if not messages:
        return ""
    return str(messages[-1].content)
