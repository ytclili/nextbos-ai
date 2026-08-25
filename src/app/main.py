from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.tracing import configure_tracing, shutdown_tracing
from app.memory.long_term.store import postgres_memory_store
from app.memory.short_term.checkpointer import redis_checkpointer
from app.persistence.postgres.database import create_engine, initialize_agent_schema
from app.persistence.postgres.session import create_session_factory


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    configure_tracing(app, settings)

    engine = create_engine(settings)
    await initialize_agent_schema(engine)
    session_factory = create_session_factory(engine=engine)

    async with (
        redis_checkpointer(settings) as checkpointer,
        postgres_memory_store(settings) as memory_store,
    ):
        app.state.settings = settings
        app.state.db_engine = engine
        app.state.session_factory = session_factory
        app.state.checkpointer = checkpointer
        app.state.memory_store = memory_store
        try:
            yield
        finally:
            await engine.dispose()
            shutdown_tracing()


app = FastAPI(title=get_settings().app_name, lifespan=lifespan)
app.include_router(router)