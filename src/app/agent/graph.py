from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from app.agent.model_runtime import AgentModelRuntime
from app.agent.nodes.respond import create_respond_node
from app.agent.state import AgentState


def fallback_respond(state: AgentState) -> dict[str, list[AIMessage]]:
    """没有注入模型运行时时使用的兜底回显节点。

    这个节点只用于测试或未完成依赖注入时保持 starter graph 可运行。
    真正调用模型时，build_graph 需要传入 model_runtime。
    """

    incoming = state.get("messages", [])[-1].content if state.get("messages") else ""
    return {"messages": [AIMessage(content=f"已收到：{incoming}")]}


def build_graph(*, checkpointer=None, model_runtime: AgentModelRuntime | None = None):
    builder = StateGraph(AgentState)

    respond_node = (
        create_respond_node(model_runtime=model_runtime)
        if model_runtime is not None
        else fallback_respond
    )

    builder.add_node("respond", respond_node)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
