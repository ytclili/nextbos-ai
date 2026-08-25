import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage

from app.agent.model_runtime import AgentModelRuntime
from app.agent.prompts.renderer import prepend_system_prompt
from app.agent.state import AgentState
from app.core.tracing import get_tracer, set_span_attributes
from app.tools.registry import get_builtin_tools

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

RespondNode = Callable[[AgentState], Awaitable[dict[str, Any]]]


def create_respond_node(
    *,
    model_runtime: AgentModelRuntime,
    tools_enabled: bool = True,
    prefer_summarized_messages: bool = True,
) -> RespondNode:
    """创建 LangGraph 的 respond 节点。

    respond 节点负责根据当前 messages 生成 assistant 回复。
    模型配置解析交给 AgentModelRuntime；模型调用、tool binding 使用 LangChain
    原生 ChatModel 接口，避免丢失 AIMessage.tool_calls。

    如果上游 summary 节点生成了 summarized_messages，则 respond 会优先使用
    summarized_messages 作为本次 LLM 输入；否则回退到原始 messages。

    tools_enabled 用来区分两类模型节点：
    - True：第一次 respond，允许模型根据需要发起工具调用；
    - False：工具执行后的 final respond，不再绑定工具，只负责生成最终文本。

    prefer_summarized_messages 控制模型输入来源：
    - True：优先使用 summary 节点产出的 summarized_messages，避免长上下文撑爆窗口；
    - False：强制使用原始 messages，确保工具执行后的 ToolMessage 能被最终回复看到。

    这样可以避免部分模型在“请记住”这类请求里反复调用 manage_memory，
    导致 respond -> tools -> respond 无限循环。
    """

    async def respond(state: AgentState) -> dict[str, Any]:
        """调用大模型生成回复，并把结果追加到 messages。"""

        with tracer.start_as_current_span("agent.node.respond") as span:
            raw_messages = state.get("messages", [])
            if prefer_summarized_messages:
                messages = state.get("summarized_messages") or raw_messages
            else:
                messages = raw_messages
            model_messages = prepend_system_prompt(messages)

            span.set_attribute("agent.node.name", "respond")
            span.set_attribute("agent.messages.count", len(raw_messages))
            span.set_attribute("agent.summarized_messages.count", len(messages))
            span.set_attribute("agent.model_messages.count", len(model_messages))
            span.set_attribute("agent.thread_id", state.get("thread_id", ""))
            span.set_attribute("agent.user_id", state.get("user_id", ""))

            config = await model_runtime.resolve_config(state.get("model_options"))
            chat_model = model_runtime.create_chat_model(config)
            tools = get_builtin_tools() if tools_enabled else []
            if tools:
                chat_model = chat_model.bind_tools(tools)
            tool_names = [tool.name for tool in tools]

            with tracer.start_as_current_span("llm.chat") as llm_span:
                llm_span.set_attribute("llm.config.source", config.source)
                llm_span.set_attribute(
                    "llm.config.snapshot_id",
                    str(config.snapshot_id) if config.snapshot_id else "",
                )
                llm_span.set_attribute("llm.provider", config.provider)
                llm_span.set_attribute("llm.model_name", config.model_name)
                llm_span.set_attribute("llm.config.digest", config.digest)
                llm_span.set_attribute("llm.messages.count", len(model_messages))
                llm_span.set_attribute("llm.tools.count", len(tools))

                logger.info(
                    (
                        "llm.chat.request provider=%s base_url=%s model_name=%s "
                        "api_key_set=%s messages_count=%s tools_count=%s tool_names=%s "
                        "thread_id=%s user_id=%s"
                    ),
                    config.provider,
                    config.base_url,
                    config.model_name,
                    bool(config.credential and config.credential.api_key),
                    len(model_messages),
                    len(tools),
                    tool_names,
                    state.get("thread_id", ""),
                    state.get("user_id", ""),
                )

                try:
                    response = await chat_model.ainvoke(model_messages)
                except Exception as exc:
                    logger.exception(
                        (
                            "llm.chat.failed provider=%s base_url=%s model_name=%s "
                            "messages_count=%s tools_count=%s tool_names=%s "
                            "thread_id=%s user_id=%s error_type=%s status_code=%s "
                            "response_body=%s"
                        ),
                        config.provider,
                        config.base_url,
                        config.model_name,
                        len(model_messages),
                        len(tools),
                        tool_names,
                        state.get("thread_id", ""),
                        state.get("user_id", ""),
                        type(exc).__name__,
                        _error_status_code(exc),
                        _short_text(_error_response_text(exc)),
                    )
                    raise

                if not isinstance(response, AIMessage):
                    raise TypeError("LangChain chat model returned a non-AIMessage response")

                content_length = len(str(response.content))
                llm_span.set_attribute("llm.response.content_length", content_length)
                set_span_attributes(llm_span, "llm.usage", dict(response.usage_metadata or {}))

            span.set_attribute("agent.response.content_length", len(str(response.content)))
            span.set_attribute("agent.response.tool_calls.count", len(response.tool_calls or []))
            return {
                "messages": [response],
                "llm_snapshot_id": config.snapshot_id,
            }

    return respond


def _error_status_code(exc: Exception) -> object:
    """从供应商异常里尽量提取 HTTP 状态码。"""

    response = getattr(exc, "response", None)
    return getattr(response, "status_code", "")


def _error_response_text(exc: Exception) -> str:
    """从供应商异常里尽量提取响应体，方便排查 400。"""

    response = getattr(exc, "response", None)
    text = getattr(response, "text", "")
    if text:
        return str(text)

    body = getattr(exc, "body", "")
    if body:
        return str(body)

    return str(exc)


def _short_text(value: str, *, max_length: int = 2000) -> str:
    """把响应体截断后写日志，避免控制台被大响应刷屏。"""

    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}...(truncated, length={len(value)})"
