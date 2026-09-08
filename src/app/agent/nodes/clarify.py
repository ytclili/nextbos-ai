from langchain_core.messages import AIMessage

from app.agent.state import AgentState


def clarify(state: AgentState) -> dict[str, list[AIMessage]]:
    """澄清分支节点，优先使用意图识别给出的澄清问题。"""
    decision = state.get("intent_decision") or {}
    question = decision.get("clarification_question") or "请补充一下你想查询的时间范围或业务条件。"
    return {"messages": [AIMessage(content=question)]}
