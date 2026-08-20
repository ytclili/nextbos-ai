from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agent.graph import build_graph


async def run_graph(
    checkpointer: BaseCheckpointSaver,
    *,
    thread_id: str,
    user_id: str,
    message: str,
):
    runnable = build_graph(checkpointer=checkpointer)
    return await runnable.ainvoke(
        {
            "messages": [HumanMessage(content=message)],
            "thread_id": thread_id,
            "user_id": user_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
