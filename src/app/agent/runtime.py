from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.graph import build_graph
from app.agent.model_runtime import AgentModelRuntime
from app.agent.nodes.summarize import SummaryOptions
from app.agent.options import ChatModelOptions
from app.conversation.context_loader import ConversationContextLoader
from app.conversation.repository import ConversationRepository
from app.core.config import Settings
from app.core.tracing import get_tracer
from app.llm.config_resolver import ModelConfigResolver
from app.persistence.postgres.llm_model_repository import PostgresLLMModelRepository

tracer = get_tracer(__name__)


async def run_graph(
    checkpointer: BaseCheckpointSaver,
    *,
    thread_id: str,
    user_id: str,
    message: str,
    model_options: ChatModelOptions,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    memory_store=None,
):
    """执行一次 LangGraph chat。

    这里负责组装 agent 本次运行需要的运行时依赖：
    DB repository -> ModelConfigResolver -> AgentModelRuntime。

    memory_store 是 LangGraph 官方 Store，给长期记忆工具使用。
    summarization_model 是 LangMem 官方 SummarizationNode 使用的摘要模型。

    如果 Redis checkpoint 还在，就让 LangGraph 按 thread_id 自动续上短期上下文；
    如果 Redis checkpoint 已经过期，就从 PostgreSQL conversation 记录恢复最近上下文。
    """

    with tracer.start_as_current_span("agent.run") as span:
        span.set_attribute("agent.thread_id", thread_id)
        span.set_attribute("agent.user_id", user_id)
        span.set_attribute("agent.input_length", len(message))

        graph_config = {
            "configurable": {
                "thread_id": thread_id,
                "langgraph_user_id": user_id,
            }
        }
        checkpoint_exists = await _checkpoint_exists(checkpointer, graph_config)
        span.set_attribute("agent.checkpoint.exists", checkpoint_exists)

        async with session_factory() as session:
            model_repository = PostgresLLMModelRepository(session)
            model_runtime = AgentModelRuntime(
                config_resolver=ModelConfigResolver(model_repository, settings),
            )
            model_config = await model_runtime.resolve_config(model_options)
            summarization_model = model_runtime.create_chat_model(model_config)

            if checkpoint_exists:
                messages = [HumanMessage(content=message)]
            else:
                context_loader = ConversationContextLoader(
                    ConversationRepository(session_factory),
                )
                messages = await context_loader.load_messages(thread_id=thread_id)
                if not messages:
                    messages = [HumanMessage(content=message)]

            span.set_attribute("agent.restore.messages.count", len(messages))

            runnable = build_graph(
                checkpointer=checkpointer,
                model_runtime=model_runtime,
                store=memory_store,
                summarization_model=summarization_model,
                summary_options=SummaryOptions(
                    max_tokens=settings.summary_max_tokens,
                    trigger_tokens=settings.summary_trigger_tokens,
                    max_output_tokens=settings.summary_max_output_tokens,
                ),
            )
            return await runnable.ainvoke(
                {
                    "messages": messages,
                    "thread_id": thread_id,
                    "user_id": user_id,
                    "model_options": model_options,
                },
                config=graph_config,
            )


async def _checkpoint_exists(checkpointer: BaseCheckpointSaver, config: dict) -> bool:
    """判断当前 thread_id 是否已经有 LangGraph checkpoint。

    Redis checkpoint 存在时，LangGraph 会自己从 checkpoint 续上下文；
    Redis checkpoint 不存在时，runtime 才需要从 PostgreSQL conversation 恢复。
    """

    checkpoint = await checkpointer.aget_tuple(config)
    return checkpoint is not None
