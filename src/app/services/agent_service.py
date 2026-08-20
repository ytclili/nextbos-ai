from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agent.runtime import run_graph


class AgentService:
    def __init__(self, checkpointer: BaseCheckpointSaver):
        self.checkpointer = checkpointer

    async def chat(self, *, thread_id: str, user_id: str, message: str) -> str:
        result = await run_graph(
            self.checkpointer,
            thread_id=thread_id,
            user_id=user_id,
            message=message,
        )
        return result["messages"][-1].content
