from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.store.base import BaseStore

from app.agent.model_runtime import AgentModelRuntime
from app.agent.nodes.respond import create_respond_node
from app.agent.nodes.summarize import SummaryOptions, create_summarize_node
from app.agent.nodes.tools import create_tools_node
from app.agent.state import AgentState


def fallback_respond(state: AgentState) -> dict[str, list[AIMessage]]:
    """没有注入模型运行时时使用的兜底回显节点。

    这个节点只用于测试或未完成依赖注入时保持 starter graph 可运行。
    真正调用模型时，build_graph 需要传入 model_runtime。
    """

    incoming = state.get("messages", [])[-1].content if state.get("messages") else ""
    return {"messages": [AIMessage(content=f"已收到：{incoming}")]}


def build_graph(
    *,
    checkpointer=None,
    model_runtime: AgentModelRuntime | None = None,
    store: BaseStore | None = None,
    summarization_model: Any | None = None,
    summary_options: SummaryOptions | None = None,
):
    """构建 agent 执行图。

    图结构：

    summarize -> respond -> tools -> final_respond -> END

    summarize 节点负责用 LangMem 官方 SummarizationNode 压缩过长上下文；
    respond 节点负责调用模型；
    tools 节点负责执行模型返回的 tool_calls。
    final_respond 节点负责在工具执行后生成最终文本，并且不再绑定工具，
    避免模型反复调用同一个工具形成循环。

    store 是 LangGraph 官方长期记忆 Store。
    这里不自己实现长期记忆，只把 Store 交给 LangGraph runtime。
    后续工具可以通过官方机制访问这个 Store。
    """

    builder = StateGraph(AgentState)

    respond_node = (
        create_respond_node(model_runtime=model_runtime)
        if model_runtime is not None
        else fallback_respond
    )
    final_respond_node = (
        create_respond_node(
            model_runtime=model_runtime,
            tools_enabled=False,
            prefer_summarized_messages=False,
        )
        if model_runtime is not None
        else fallback_respond
    )

    builder.add_node(
        "summarize",
        create_summarize_node(summarization_model, options=summary_options),
    )
    builder.add_node("respond", respond_node)
    builder.add_node("tools", create_tools_node())
    builder.add_node("final_respond", final_respond_node)

    builder.add_edge(START, "summarize")
    builder.add_edge("summarize", "respond")
    builder.add_conditional_edges(
        "respond",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )
    builder.add_edge("tools", "final_respond")
    builder.add_edge("final_respond", END)

    return builder.compile(checkpointer=checkpointer, store=store)


graph = build_graph()
