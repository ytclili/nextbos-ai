from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command
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
from app.tools.business.auth import use_core_internal_token

tracer = get_tracer(__name__)


@dataclass(frozen=True)
class PreparedGraphRun:
    """一次 LangGraph 运行前准备好的公共上下文。"""

    runnable: Any
    input_state: dict[str, Any]
    graph_config: dict[str, Any]


@dataclass(frozen=True)
class GraphStreamEvent:
    """LangGraph 流式事件。

    mode 是 LangGraph stream_mode，例如 messages / updates / final_state。
    data 是该模式下的原始数据，service 层再决定如何转成前端事件。
    """

    mode: str
    data: Any


async def run_graph(
    checkpointer: BaseCheckpointSaver,
    *,
    thread_id: str,
    user_id: str,
    message: str,
    model_options: ChatModelOptions,
    token: str | None = None,
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

        async with _prepare_graph_run(
            checkpointer,
            thread_id=thread_id,
            user_id=user_id,
            message=message,
            token=token,
            model_options=model_options,
            session_factory=session_factory,
            settings=settings,
            memory_store=memory_store,
        ) as prepared:
            span.set_attribute(
                "agent.restore.messages.count",
                len(prepared.input_state["messages"]),
            )
            token_override = use_core_internal_token(token)
            try:
                return await prepared.runnable.ainvoke(
                    prepared.input_state,
                    config=prepared.graph_config,
                )
            finally:
                token_override.reset()


async def stream_graph(
    checkpointer: BaseCheckpointSaver,
    *,
    thread_id: str,
    user_id: str,
    message: str,
    model_options: ChatModelOptions,
    token: str | None = None,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    memory_store=None,
) -> AsyncIterator[GraphStreamEvent]:
    """流式执行一次 LangGraph chat。

    这里复用和 run_graph 相同的图准备逻辑，只是最后改为调用 LangGraph
    官方 astream()。stream_mode 同时打开：
    - messages：拿 LLM token 增量；
    - updates：保留后续扩展节点/tool 状态事件的入口。

    图执行结束后再读取最终 state，service 层用它保存 assistant 完整回复和 summary。
    """

    with tracer.start_as_current_span("agent.run.stream") as span:
        span.set_attribute("agent.thread_id", thread_id)
        span.set_attribute("agent.user_id", user_id)
        span.set_attribute("agent.input_length", len(message))

        async with _prepare_graph_run(
            checkpointer,
            thread_id=thread_id,
            user_id=user_id,
            message=message,
            token=token,
            model_options=model_options,
            session_factory=session_factory,
            settings=settings,
            memory_store=memory_store,
        ) as prepared:
            span.set_attribute(
                "agent.restore.messages.count",
                len(prepared.input_state["messages"]),
            )

            token_override = use_core_internal_token(token)
            try:
                async for item in prepared.runnable.astream(
                    prepared.input_state,
                    config=prepared.graph_config,
                    stream_mode=["messages", "updates"],
                ):
                    yield _to_graph_stream_event(item)

                snapshot = await prepared.runnable.aget_state(prepared.graph_config)
                yield GraphStreamEvent(
                    mode="final_state",
                    data=dict(getattr(snapshot, "values", {}) or {}),
                )
            finally:
                token_override.reset()


async def stream_graph_resume(
    checkpointer: BaseCheckpointSaver,
    *,
    thread_id: str,
    user_id: str,
    resume: dict[str, Any],
    model_options: ChatModelOptions,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    memory_store=None,
) -> AsyncIterator[GraphStreamEvent]:
    """从 LangGraph interrupt checkpoint 恢复一次流式运行。"""

    with tracer.start_as_current_span("agent.run.stream.resume") as span:
        span.set_attribute("agent.thread_id", thread_id)
        span.set_attribute("agent.user_id", user_id)

        async with _prepare_graph_run(
            checkpointer,
            thread_id=thread_id,
            user_id=user_id,
            message="",
            model_options=model_options,
            session_factory=session_factory,
            settings=settings,
            memory_store=memory_store,
            resume=True,
        ) as prepared:
            graph_config = _with_resume_config(prepared.graph_config, resume)
            sanitized_resume = _resume_without_token(resume)
            token_override = use_core_internal_token(_resume_token(resume))
            try:
                async for item in prepared.runnable.astream(
                    Command(resume=sanitized_resume),
                    config=graph_config,
                    stream_mode=["messages", "updates"],
                ):
                    yield _to_graph_stream_event(item)

                snapshot = await prepared.runnable.aget_state(graph_config)
                yield GraphStreamEvent(
                    mode="final_state",
                    data=dict(getattr(snapshot, "values", {}) or {}),
                )
            finally:
                token_override.reset()


@asynccontextmanager
async def _prepare_graph_run(
    checkpointer: BaseCheckpointSaver,
    *,
    thread_id: str,
    user_id: str,
    message: str,
    model_options: ChatModelOptions,
    token: str | None = None,
    session_factory: async_sessionmaker[AsyncSession],
    settings: Settings,
    memory_store=None,
    resume: bool = False,
) -> AsyncIterator[PreparedGraphRun]:
    """准备 LangGraph 运行所需的公共上下文。

    非流式 run_graph 和流式 stream_graph 都走这里，避免两套逻辑漂移。

    注意：这里是 async contextmanager，因为 AgentModelRuntime 里的模型仓储会持有
    本次数据库 session。必须等图执行结束后再退出 session context。
    """

    graph_config = {
        "configurable": {
            "thread_id": thread_id,
            "langgraph_user_id": user_id,
        }
    }
    checkpoint_exists = await _checkpoint_exists(checkpointer, graph_config)

    async with session_factory() as session:
        model_repository = PostgresLLMModelRepository(session)
        model_runtime = AgentModelRuntime(
            config_resolver=ModelConfigResolver(model_repository, settings),
        )
        model_config = await model_runtime.resolve_config(model_options)
        summarization_model = model_runtime.create_chat_model(model_config)

        if resume:
            messages = []
        elif checkpoint_exists:
            messages = [HumanMessage(content=message)]
        else:
            context_loader = ConversationContextLoader(
                ConversationRepository(session_factory),
            )
            messages = await context_loader.load_messages(thread_id=thread_id)
            if not messages:
                messages = [HumanMessage(content=message)]

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
        yield PreparedGraphRun(
            runnable=runnable,
            input_state={
                "messages": messages,
                "thread_id": thread_id,
                "user_id": user_id,
                "model_options": model_options,
            },
            graph_config=graph_config,
        )


def _to_graph_stream_event(item: Any) -> GraphStreamEvent:
    """把 LangGraph astream 原始返回规整成 GraphStreamEvent。"""

    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
        return GraphStreamEvent(mode=item[0], data=item[1])
    return GraphStreamEvent(mode="updates", data=item)


def _with_resume_config(graph_config: dict[str, Any], resume: dict[str, Any]) -> dict[str, Any]:
    """把前端 resume 数据放进 LangGraph config，供节点读取。"""

    configurable = dict(graph_config.get("configurable") or {})
    configurable["auth_resume"] = _resume_without_token(resume)

    return {**graph_config, "configurable": configurable}


def _resume_without_token(resume: dict[str, Any]) -> dict[str, Any]:
    """移除 resume payload 里的 token，避免写入 LangGraph checkpoint/config。"""

    return {key: value for key, value in resume.items() if key != "token"}


def _resume_token(resume: dict[str, Any]) -> str | None:
    """读取前端 resume payload 里的 token。"""

    token = resume.get("token")
    return token.strip() if isinstance(token, str) and token.strip() else None


async def _checkpoint_exists(checkpointer: BaseCheckpointSaver, config: dict) -> bool:
    """判断当前 thread_id 是否已经有 LangGraph checkpoint。

    Redis checkpoint 存在时，LangGraph 会自己从 checkpoint 续上下文；
    Redis checkpoint 不存在时，runtime 才需要从 PostgreSQL conversation 恢复。
    """

    checkpoint = await checkpointer.aget_tuple(config)
    return checkpoint is not None
