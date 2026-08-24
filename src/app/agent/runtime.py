from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.graph import build_graph
from app.agent.model_runtime import AgentModelRuntime
from app.agent.options import ChatModelOptions
from app.core.config import Settings
from app.llm.config_resolver import ModelConfigResolver
from app.llm.service import LLMService
from app.persistence.postgres.llm_model_repository import PostgresLLMModelRepository


async def run_graph(
    checkpointer: BaseCheckpointSaver,
    *,
    thread_id: str,
    user_id: str,
    message: str,
    model_options: ChatModelOptions,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
):
    """执行一次 LangGraph chat。

    这里负责组装 agent 本次运行需要的模型运行时依赖：
    DB repository -> ModelConfigResolver -> LLMService -> AgentModelRuntime。
    """

    async with session_factory() as session:
        model_repository = PostgresLLMModelRepository(session)
        model_runtime = AgentModelRuntime(
            config_resolver=ModelConfigResolver(model_repository, settings),
            llm_service=LLMService(),
        )

        runnable = build_graph(checkpointer=checkpointer, model_runtime=model_runtime)
        return await runnable.ainvoke(
            {
                "messages": [HumanMessage(content=message)],
                "thread_id": thread_id,
                "user_id": user_id,
                "model_options": model_options,
            },
            config={"configurable": {"thread_id": thread_id}},
        )
