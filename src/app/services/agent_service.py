import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.options import ChatModelOptions
from app.agent.runtime import run_graph
from app.conversation.repository import ConversationRepository
from app.conversation.summary import extract_running_summary
from app.core.config import Settings

logger = logging.getLogger(__name__)


class AgentService:
    """chat 接口到 LangGraph runtime 之间的应用服务层。"""

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        memory_store=None,
    ):
        self.checkpointer = checkpointer
        self.session_factory = session_factory
        self.settings = settings
        self.memory_store = memory_store
        self.conversation_repository = ConversationRepository(session_factory)

    async def chat(
        self,
        *,
        thread_id: str,
        user_id: str,
        message: str,
        model_options: ChatModelOptions | None = None,
        trace_id: str | None = None,
    ) -> str:
        """执行一次 chat。

        trace_id 会写入 conversation_messages。
        这样后续可以从聊天记录反查 SigNoZ / OpenTelemetry 调用链路。
        """

        logger.info(
            "agent.chat.started thread_id=%s user_id=%s input_length=%s",
            thread_id,
            user_id,
            len(message),
        )

        # 先保存用户消息。
        # 这是一个短事务，提交后马上释放数据库连接，不会跨 LLM 调用持有事务。
        await self.conversation_repository.append_user_message(
            thread_id=thread_id,
            user_id=user_id,
            content=message,
            trace_id=trace_id,
        )

        result = await run_graph(
            self.checkpointer,
            thread_id=thread_id,
            user_id=user_id,
            message=message,
            model_options=model_options or ChatModelOptions(),
            session_factory=self.session_factory,
            settings=self.settings,
            memory_store=self.memory_store,
        )

        content = result["messages"][-1].content
        llm_snapshot_id = result.get("llm_snapshot_id")

        # 再保存 assistant 最终回复。
        # trace_id 和 user 消息保持一致，llm_snapshot_id 指向本次回答实际使用的模型快照。
        await self.conversation_repository.append_assistant_message(
            thread_id=thread_id,
            user_id=user_id,
            content=content,
            trace_id=trace_id,
            llm_snapshot_id=llm_snapshot_id,
        )

        # 如果 LangMem summary 节点产出了 rolling summary，就把它持久化。
        # 这仍然是短事务，不会跨 LLM 调用持有数据库事务。
        if summary := extract_running_summary(result):
            await self.conversation_repository.save_summary(
                thread_id=thread_id,
                user_id=user_id,
                summary=summary.summary,
                covered_through_message_id=summary.covered_through_message_id,
                message_count=summary.message_count,
            )

        logger.info(
            "agent.chat.completed thread_id=%s user_id=%s output_length=%s",
            thread_id,
            user_id,
            len(content),
        )

        return content