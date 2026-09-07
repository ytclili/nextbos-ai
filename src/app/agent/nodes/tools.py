import logging
from collections.abc import Awaitable, Callable
from time import perf_counter

from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from app.agent.state import AgentState
from app.core.tracing import get_tracer
from app.tools.registry import get_builtin_tools

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

ToolsNode = Callable[[AgentState], Awaitable[dict]]


def create_tools_node() -> ToolsNode:
    """创建 LangGraph 工具执行节点。

    这个节点负责执行 AIMessage 里的 tool_calls。
    工具本身由 LangChain @tool 定义；
    这里不手写执行逻辑，仍然复用 LangGraph 官方 ToolNode。

    外层只补充项目自己的 trace/log，方便在 SigNoZ 里看到：
    - 调用了哪个工具；
    - 工具调用是否成功；
    - 工具耗时多少。
    """

    tool_node = ToolNode(get_builtin_tools())

    async def tools(state: AgentState) -> dict:
        """执行模型请求的工具调用。"""

        thread_id = state.get("thread_id", "")
        user_id = state.get("user_id", "")
        tool_calls = _latest_tool_calls(state)
        started_at = perf_counter()

        with tracer.start_as_current_span("agent.node.tools") as span:
            span.set_attribute("agent.node.name", "tools")
            span.set_attribute("agent.thread_id", thread_id)
            span.set_attribute("agent.user_id", user_id)
            span.set_attribute("agent.tool.calls.count", len(tool_calls))

            for tool_call in tool_calls:
                tool_name = str(tool_call.get("name", ""))
                tool_call_id = str(tool_call.get("id", ""))
                span.add_event(
                    "agent.tool.started",
                    {
                        "agent.tool.name": tool_name,
                        "agent.tool.call_id": tool_call_id,
                    },
                )
                logger.info(
                    "agent.tool.started tool_name=%s tool_call_id=%s thread_id=%s user_id=%s",
                    tool_name,
                    tool_call_id,
                    thread_id,
                    user_id,
                )

            try:
                result = await tool_node.ainvoke(state)
            except Exception:
                duration_ms = _duration_ms(started_at)
                span.set_attribute("agent.tool.status", "error")
                span.set_attribute("agent.tool.duration_ms", duration_ms)
                for tool_call in tool_calls:
                    tool_name = str(tool_call.get("name", ""))
                    tool_call_id = str(tool_call.get("id", ""))
                    span.add_event(
                        "agent.tool.failed",
                        {
                            "agent.tool.name": tool_name,
                            "agent.tool.call_id": tool_call_id,
                            "agent.tool.duration_ms": duration_ms,
                        },
                    )
                    logger.exception(
                        (
                            "agent.tool.failed tool_name=%s tool_call_id=%s "
                            "thread_id=%s user_id=%s duration_ms=%s"
                        ),
                        tool_name,
                        tool_call_id,
                        thread_id,
                        user_id,
                        duration_ms,
                    )
                raise

            duration_ms = _duration_ms(started_at)
            span.set_attribute("agent.tool.status", "success")
            span.set_attribute("agent.tool.duration_ms", duration_ms)
            for tool_call in tool_calls:
                tool_name = str(tool_call.get("name", ""))
                tool_call_id = str(tool_call.get("id", ""))
                span.add_event(
                    "agent.tool.completed",
                    {
                        "agent.tool.name": tool_name,
                        "agent.tool.call_id": tool_call_id,
                        "agent.tool.duration_ms": duration_ms,
                    },
                )
                logger.info(
                    (
                        "agent.tool.completed tool_name=%s tool_call_id=%s "
                        "thread_id=%s user_id=%s duration_ms=%s"
                    ),
                    tool_name,
                    tool_call_id,
                    thread_id,
                    user_id,
                    duration_ms,
                )

            return result

    return tools


def _latest_tool_calls(state: AgentState) -> list[dict]:
    """读取最近一条 AIMessage 上的 tool_calls。"""

    messages = state.get("messages", [])
    if not messages:
        return []

    latest_message = messages[-1]
    if not isinstance(latest_message, AIMessage):
        return []

    return list(latest_message.tool_calls or [])


def _duration_ms(started_at: float) -> int:
    """计算毫秒耗时。"""

    return int((perf_counter() - started_at) * 1000)
