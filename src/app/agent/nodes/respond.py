from collections.abc import Awaitable, Callable

from langchain_core.messages import AIMessage

from app.agent.model_runtime import AgentModelRuntime
from app.agent.state import AgentState

RespondNode = Callable[[AgentState], Awaitable[dict[str, list[AIMessage]]]]


def create_respond_node(*, model_runtime: AgentModelRuntime) -> RespondNode:
    """创建 LangGraph 的 respond 节点。

    respond 节点负责根据当前 messages 生成 assistant 回复。
    具体模型配置解析和 LLM 调用交给 AgentModelRuntime。
    """

    async def respond(state: AgentState) -> dict[str, list[AIMessage]]:
        """调用大模型生成回复，并把结果追加到 messages。"""

        completion = await model_runtime.chat(
            messages=state.get("messages", []),
            options=state.get("model_options"),
        )

        return {"messages": [AIMessage(content=completion.content)]}

    return respond
