from langchain_core.messages import AIMessage

from app.agent.state import AgentState


def wren_context_stub(state: AgentState) -> dict[str, list[AIMessage]]:
    """业务数据分支占位节点，后续会替换为 WrenAI 上下文查询。"""

    return {
        "messages": [
            AIMessage(content="已识别为业务数据问题，下一步将接入 WrenAI 上下文查询。")
        ]
    }
