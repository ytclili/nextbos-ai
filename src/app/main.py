from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.memory.short_term.checkpointer import redis_checkpointer
from app.persistence.postgres.database import create_engine, initialize_agent_schema


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    # Postgres：服务启动时建好 engine，挂到 app.state 供后续功能复用
    engine = create_engine(settings)
    await initialize_agent_schema(engine)
    app.state.engine = engine

    try:
        # Redis checkpointer：连接 + setup()（建索引）只在这里执行一次
        async with redis_checkpointer(settings) as checkpointer:
            app.state.checkpointer = checkpointer
            yield  # ← 整个服务生命周期，所有请求共用上面这两样东西
    finally:
        await engine.dispose()


app = FastAPI(title=get_settings().app_name, lifespan=lifespan)
app.include_router(router)
