import asyncio
from collections.abc import Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.graph import build_graph
from app.agent.model_runtime import AgentModelRuntime
from app.agent.options import ChatModelOptions
from app.core.config import Settings, get_settings
from app.llm.config_resolver import ModelConfigResolver
from app.llm.ports import ChatCompletion
from app.llm.service import LLMService
from app.persistence.postgres.database import create_engine, initialize_agent_schema
from app.persistence.postgres.llm_model_repository import PostgresLLMModelRepository
from app.persistence.postgres.session import create_session_factory


class StudioModelRuntime:
    """LangGraph Studio 使用的模型运行时。

    FastAPI 入口会在 app lifespan 里创建 engine、session_factory、settings。
    但 LangGraph Studio 直接从 langgraph.json 导入 graph，不会经过 FastAPI lifespan，
    所以这里单独为 Studio 准备一份最小运行时装配。
    """

    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        initialize_schema: Callable[[], Awaitable[None]],
    ):
        self.settings = settings
        self.session_factory = session_factory
        self.initialize_schema = initialize_schema
        self._schema_initialized = False
        self._schema_lock = asyncio.Lock()

    async def chat(
        self,
        *,
        messages: list[object],
        options: ChatModelOptions | None,
    ) -> ChatCompletion:
        """在 Studio 中调用真实模型链路。"""

        await self._ensure_schema()

        async with self.session_factory() as session:
            model_repository = PostgresLLMModelRepository(session)
            model_runtime = AgentModelRuntime(
                config_resolver=ModelConfigResolver(model_repository, self.settings),
                llm_service=LLMService(),
            )
            return await model_runtime.chat(messages=messages, options=options)

    async def _ensure_schema(self) -> None:
        """首次 Studio 调用前初始化数据库表结构。"""

        if self._schema_initialized:
            return

        async with self._schema_lock:
            if self._schema_initialized:
                return
            await self.initialize_schema()
            self._schema_initialized = True


def create_studio_graph():
    """创建给 LangGraph Studio 使用的 graph。

    这个入口会注入真实 StudioModelRuntime，因此 Studio 里直接运行也会调用模型。
    """

    settings = get_settings()
    engine = create_engine(settings)
    session_factory = create_session_factory(engine=engine)

    async def initialize_schema() -> None:
        await initialize_agent_schema(engine)

    model_runtime = StudioModelRuntime(
        settings=settings,
        session_factory=session_factory,
        initialize_schema=initialize_schema,
    )
    return build_graph(model_runtime=model_runtime)


graph = create_studio_graph()
