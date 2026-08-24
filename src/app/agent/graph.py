from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from app.agent.model_runtime import AgentModelRuntime
from app.agent.nodes.respond import create_respond_node
from app.agent.nodes.tools import create_tools_node
from app.agent.state import AgentState


def fallback_respond(state: AgentState) -> dict[str, list[AIMessage]]:
    """没有注入模型运行时时使用的兜底回显节点。

    这个节点只用于测试或未完成依赖注入时保持 starter graph 可运行。
    真正调用模型时，build_graph 需要传入 model_runtime。
    """

    incoming = state.get("messages", [])[-1].content if state.get("messages") else ""
    return {"messages": [AIMessage(content=f"已收到：{incoming}")]}


def build_graph(*, checkpointer=None, model_runtime: AgentModelRuntime | None = None):
    """构建 agent 执行图。

    图结构：

    respond -> tools -> respond -> END

    respond 节点负责调用模型。
    如果模型返回 tool_calls，则进入 tools 节点执行工具；
    tools 节点执行完会把 ToolMessage 写回 messages，然后回到 respond。
    如果模型没有返回 tool_calls，则直接结束。
    """

    builder = StateGraph(AgentState)

    respond_node = (
        create_respond_node(model_runtime=model_runtime)
        if model_runtime is not None
        else fallback_respond
    )

    builder.add_node("respond", respond_node)
    builder.add_node("tools", create_tools_node())

    builder.add_edge(START, "respond")
    builder.add_conditional_edges(
        "respond",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )
    builder.add_edge("tools", "respond")

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()