from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.store.base import BaseStore

from app.agent.model_runtime import AgentModelRuntime
from app.agent.nodes.chart_plan import create_chart_plan_node
from app.agent.nodes.chart_spec import create_chart_spec_node
from app.agent.nodes.clarify import clarify
from app.agent.nodes.intent import create_intent_node
from app.agent.nodes.respond import create_respond_node
from app.agent.nodes.route import (
    route_after_intent,
    route_after_sql_execute,
    route_after_sql_plan,
)
from app.agent.nodes.sql_execute import create_sql_execute_node
from app.agent.nodes.sql_plan import create_sql_plan_node
from app.agent.nodes.sql_validate import create_sql_validate_node
from app.agent.nodes.summarize import SummaryOptions, create_summarize_node
from app.agent.nodes.tools import create_tools_node
from app.agent.nodes.wren_context import create_wren_context_node
from app.agent.state import AgentState
from app.integrations.business.sql_executor import SqlExecutionClient
from app.integrations.business.wren import WrenContextClient


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
    wren_context_client: WrenContextClient | None = None,
    sql_execution_client: SqlExecutionClient | None = None,
):
    """构建 agent 执行图。

    有模型运行时时的图结构：

    summarize -> intent -> direct_answer -> respond -> tools -> final_respond -> END
                        -> wren_context -> sql_plan -> sql_validate -> sql_execute
                           -> final_respond -> END
                           -> chart_plan -> chart_spec -> final_respond -> END
                                                    -> clarify -> END
                        -> clarify -> END

    summarize 节点负责用 LangMem 官方 SummarizationNode 压缩过长上下文；
    intent 节点负责识别结构化意图，并用后端规则计算下一跳；
    wren_context 节点负责查询 WrenAI 业务上下文；
    sql_plan 节点负责基于 WrenAI 上下文生成可校验的 SQL 计划；
    sql_validate 节点负责通过 Wren dry-plan / dry-run 校验候选 SQL；
    sql_execute 节点负责执行通过校验的 SQL，并把真实数据集写入 state；
    chart_plan 节点负责生成图表规划，第一阶段先作为图表分支占位；
    chart_spec 节点负责把图表规划和真实 SQL rows 组装成 ECharts option；
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
    if model_runtime is None:
        builder.add_edge("summarize", "respond")
    else:
        builder.add_node("intent", create_intent_node(model_runtime=model_runtime))
        builder.add_node(
            "wren_context",
            create_wren_context_node(wren_context_client=wren_context_client),
        )
        builder.add_node("sql_plan", create_sql_plan_node(model_runtime=model_runtime))
        builder.add_node(
            "sql_validate",
            create_sql_validate_node(wren_context_client=wren_context_client),
        )
        builder.add_node(
            "sql_execute",
            create_sql_execute_node(sql_execution_client=sql_execution_client),
        )
        builder.add_node("chart_plan", create_chart_plan_node(model_runtime=model_runtime))
        builder.add_node("chart_spec", create_chart_spec_node())
        builder.add_node("clarify", clarify)
        builder.add_edge("summarize", "intent")
        builder.add_conditional_edges(
            "intent",
            route_after_intent,
            {
                "direct_answer": "respond",
                "wren_context": "wren_context",
                "clarify": "clarify",
            },
        )
        builder.add_edge("wren_context", "sql_plan")
        builder.add_conditional_edges(
            "sql_plan",
            route_after_sql_plan,
            {
                "sql_validate": "sql_validate",
                "clarify": "clarify",
            },
        )
        builder.add_edge("sql_validate", "sql_execute")
        builder.add_conditional_edges(
            "sql_execute",
            route_after_sql_execute,
            {
                "final_respond": "final_respond",
                "chart_plan": "chart_plan",
            },
        )
        builder.add_edge("chart_plan", "chart_spec")
        builder.add_edge("chart_spec", "final_respond")
        builder.add_edge("clarify", END)
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
