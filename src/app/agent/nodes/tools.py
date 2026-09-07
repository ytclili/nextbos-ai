import logging
from collections.abc import Awaitable, Callable
from time import perf_counter

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

from app.agent.state import AgentState
from app.core.tracing import get_tracer
from app.tools.registry import get_builtin_tools

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

ToolsNode = Callable[..., Awaitable[dict]]


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

    tool_node = ToolNode(get_builtin_tools(), handle_tool_errors=_format_tool_error)

    async def tools(state: AgentState, config: RunnableConfig | None = None) -> dict:
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
                result = await tool_node.ainvoke(state, config=config)
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
            has_tool_error = _has_tool_error(result)
            span.set_attribute("agent.tool.status", "error" if has_tool_error else "success")
            span.set_attribute("agent.tool.duration_ms", duration_ms)
            _log_tool_result_messages(result, thread_id=thread_id, user_id=user_id)
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


def _format_tool_error(exc: Exception) -> str:
    """把工具异常转换成 ToolMessage 内容，避免缺失 tool output 污染 checkpoint。

    OpenAI-compatible 接口要求每个 assistant tool_call 后面都必须有对应
    ToolMessage。这里让 ToolNode 用错误 ToolMessage 承载失败信息，后续
    final_respond 节点就可以正常告诉用户“业务接口暂时不可用”。
    """

    return f"工具执行失败：{type(exc).__name__}: {_short_text(str(exc))}"


def _has_tool_error(result: dict) -> bool:
    """判断 ToolNode 返回里是否包含失败的 ToolMessage。"""

    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not isinstance(messages, list):
        messages = [messages]

    return any(
        isinstance(message, ToolMessage)
        and (getattr(message, "status", "success") or "success") == "error"
        for message in messages
    )


def _log_tool_result_messages(result: dict, *, thread_id: str, user_id: str) -> None:
    """把 ToolNode 返回的 ToolMessage 摘要打印到控制台日志。"""

    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not isinstance(messages, list):
        messages = [messages]

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        logger.info(
            (
                "agent.tool.result tool_name=%s tool_call_id=%s status=%s "
                "thread_id=%s user_id=%s content=%s"
            ),
            message.name or "",
            message.tool_call_id or "",
            getattr(message, "status", "success") or "success",
            thread_id,
            user_id,
            _short_text(str(message.content)),
        )


def _short_text(value: str, *, max_length: int = 2000) -> str:
    """把工具返回内容截断后写日志，避免控制台被大响应刷屏。"""

    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}...(truncated, length={len(value)})"
