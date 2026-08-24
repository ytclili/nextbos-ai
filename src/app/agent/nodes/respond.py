from collections.abc import Awaitable, Callable

from langchain_core.messages import AIMessage

from app.agent.model_runtime import AgentModelRuntime
from app.agent.state import AgentState
from app.core.tracing import get_tracer, set_span_attributes
from app.tools.registry import get_builtin_tools

tracer = get_tracer(__name__)

RespondNode = Callable[[AgentState], Awaitable[dict[str, list[AIMessage]]]]


def create_respond_node(*, model_runtime: AgentModelRuntime) -> RespondNode:
    """创建 LangGraph 的 respond 节点。

    respond 节点负责根据当前 messages 生成 assistant 回复。
    模型配置解析交给 AgentModelRuntime；模型调用、tool binding 使用 LangChain
    原生 ChatModel 接口，避免丢失 AIMessage.tool_calls。
    """

    async def respond(state: AgentState) -> dict[str, list[AIMessage]]:
        """调用大模型生成回复，并把结果追加到 messages。"""

        with tracer.start_as_current_span("agent.node.respond") as span:
            messages = state.get("messages", [])
            span.set_attribute("agent.node.name", "respond")
            span.set_attribute("agent.messages.count", len(messages))
            span.set_attribute("agent.thread_id", state.get("thread_id", ""))
            span.set_attribute("agent.user_id", state.get("user_id", ""))

            config = await model_runtime.resolve_config(state.get("model_options"))
            chat_model = model_runtime.create_chat_model(config)
            tools = get_builtin_tools()
            if tools:
                chat_model = chat_model.bind_tools(tools)

            with tracer.start_as_current_span("llm.chat") as llm_span:
                llm_span.set_attribute("llm.config.source", config.source)
                llm_span.set_attribute("llm.provider", config.provider)
                llm_span.set_attribute("llm.model_name", config.model_name)
                llm_span.set_attribute("llm.config.digest", config.digest)
                llm_span.set_attribute("llm.messages.count", len(messages))
                llm_span.set_attribute("llm.tools.count", len(tools))

                response = await chat_model.ainvoke(messages)

                if not isinstance(response, AIMessage):
                    raise TypeError("LangChain chat model returned a non-AIMessage response")

                content_length = len(str(response.content))
                llm_span.set_attribute("llm.response.content_length", content_length)
                set_span_attributes(llm_span, "llm.usage", dict(response.usage_metadata or {}))

            span.set_attribute("agent.response.content_length", len(str(response.content)))
            span.set_attribute("agent.response.tool_calls.count", len(response.tool_calls or []))
            return {"messages": [response]}

    return respond
