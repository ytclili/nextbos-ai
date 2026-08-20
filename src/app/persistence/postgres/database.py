from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import Settings
from app.persistence.postgres.models import Base


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(settings.postgres_dsn, pool_pre_ping=True)


async def initialize_agent_schema(engine: AsyncEngine) -> None:
    """Create only tables owned by the agent runtime."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
