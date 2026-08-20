from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState


def respond(state: AgentState) -> dict[str, list[AIMessage]]:
    incoming = state.get("messages", [])[-1].content if state.get("messages") else ""
    return {"messages": [AIMessage(content=f"已收到：{incoming}")]}


def build_graph(*, checkpointer=None):
    builder = StateGraph(AgentState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
