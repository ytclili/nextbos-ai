import logging
import re
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

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

            auth_resume = _auth_resume_from_config(config)

            try:
                if _auth_resume_status(auth_resume) == "failed":
                    result = _auth_resume_failed_result(tool_calls, auth_resume)
                else:
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

            auth_payload = _auth_required_payload(result)
            if auth_payload is not None:
                resume = interrupt(auth_payload)
                result = _tool_auth_resume_result(result, resume)

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


def _auth_required_payload(result: dict) -> dict[str, Any] | None:
    """从工具结果里识别 401/403，并组装 LangGraph interrupt payload。"""

    for message in _tool_messages(result):
        status = getattr(message, "status", "success") or "success"
        if status != "error":
            continue

        content = str(message.content)
        if status_code := _extract_auth_status_code(content):
            return {
                "type": "auth_required",
                "name": str(message.name or ""),
                "tool_call_id": str(message.tool_call_id or ""),
                "status": "interrupted",
                "status_code": status_code,
                "message": "登录已失效或没有权限，请重新登录后再试。",
            }
    return None


def _tool_auth_resume_result(result: dict, resume: Any) -> dict:
    """根据前端授权结果，把中断恢复成可继续进入 final_respond 的工具消息。"""

    if _auth_resume_status(resume) == "failed":
        reason = _auth_resume_reason(resume) or "用户取消登录或没有权限。"
        return _replace_tool_auth_error(
            result,
            f"用户授权失败，无法继续查询业务数据。原因：{reason}",
        )

    return _replace_tool_auth_error(
        result,
        "用户已完成授权，但业务接口仍然返回无权限，请提示用户稍后重试或联系管理员。",
    )


def _auth_resume_failed_result(tool_calls: list[dict], resume: Any) -> dict:
    """前端明确授权失败时，不再重复调用业务工具，直接返回工具失败消息。"""

    reason = _auth_resume_reason(resume) or "用户取消登录或没有权限。"
    return {
        "messages": [
            ToolMessage(
                content=f"用户授权失败，无法继续查询业务数据。原因：{reason}",
                tool_call_id=str(tool_call.get("id", "")),
                name=str(tool_call.get("name", "")),
                status="error",
            )
            for tool_call in tool_calls
        ]
    }


def _replace_tool_auth_error(result: dict, content: str) -> dict:
    """把原始 401/403 工具错误替换成适合模型生成用户回复的内容。"""

    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not isinstance(messages, list):
        messages = [messages]

    replaced_messages = []
    for message in messages:
        if (
            isinstance(message, ToolMessage)
            and (getattr(message, "status", "success") or "success") == "error"
            and _extract_auth_status_code(str(message.content)) is not None
        ):
            replaced_messages.append(
                ToolMessage(
                    content=content,
                    tool_call_id=str(message.tool_call_id or ""),
                    name=str(message.name or ""),
                    status="error",
                )
            )
        else:
            replaced_messages.append(message)

    return {**result, "messages": replaced_messages}


def _auth_resume_status(resume: Any) -> str:
    """读取前端 resume payload 中的授权结果。"""

    if isinstance(resume, dict):
        return str(resume.get("status") or "")
    return ""


def _auth_resume_reason(resume: Any) -> str:
    """读取前端 resume payload 中的失败原因。"""

    if isinstance(resume, dict):
        return str(resume.get("reason") or "")
    return ""


def _auth_resume_from_config(config: RunnableConfig | None) -> Any:
    """从 LangGraph config 中读取前端 resume payload。"""

    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    return configurable.get("auth_resume")


def _extract_auth_status_code(message: str) -> int | None:
    """从工具错误文本中识别鉴权失败状态码。"""

    match = re.search(r"\b(401|403)\b", message)
    if match is None:
        return None
    return int(match.group(1))


def _has_tool_error(result: dict) -> bool:
    """判断 ToolNode 返回里是否包含失败的 ToolMessage。"""

    return any(
        isinstance(message, ToolMessage)
        and (getattr(message, "status", "success") or "success") == "error"
        for message in _tool_messages(result)
    )


def _log_tool_result_messages(result: dict, *, thread_id: str, user_id: str) -> None:
    """把 ToolNode 返回的 ToolMessage 摘要打印到控制台日志。"""

    for message in _tool_messages(result):
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


def _tool_messages(result: dict) -> list[ToolMessage]:
    """从 ToolNode 返回结果里提取 ToolMessage。"""

    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not isinstance(messages, list):
        messages = [messages]

    return [message for message in messages if isinstance(message, ToolMessage)]


def _short_text(value: str, *, max_length: int = 2000) -> str:
    """把工具返回内容截断后写日志，避免控制台被大响应刷屏。"""

    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}...(truncated, length={len(value)})"
