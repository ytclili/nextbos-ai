import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.options import ChatModelOptions
from app.agent.runtime import run_graph
from app.core.config import Settings

logger = logging.getLogger(__name__)


class AgentService:
    """chat 接口到 LangGraph runtime 之间的应用服务层。"""

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
    ):
        self.checkpointer = checkpointer
        self.session_factory = session_factory
        self.settings = settings

    async def chat(
        self,
        *,
        thread_id: str,
        user_id: str,
        message: str,
        model_options: ChatModelOptions | None = None,
    ) -> str:
        """执行一次 chat。"""

        logger.info(
            "agent.chat.started thread_id=%s user_id=%s input_length=%s",
            thread_id,
            user_id,
            len(message),
        )

        result = await run_graph(
            self.checkpointer,
            thread_id=thread_id,
            user_id=user_id,
            message=message,
            model_options=model_options or ChatModelOptions(),
            session_factory=self.session_factory,
            settings=self.settings,
        )

        content = result["messages"][-1].content
        logger.info(
            "agent.chat.completed thread_id=%s user_id=%s output_length=%s",
            thread_id,
            user_id,
            len(content),
        )

        return content
