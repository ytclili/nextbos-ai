from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.store.postgres import AsyncPostgresStore

from app.core.config import Settings


def to_psycopg_conn_string(dsn: str) -> str:
    """把 SQLAlchemy asyncpg DSN 转成 psycopg 可用的连接串。

    项目里现有 PostgreSQL 配置是 SQLAlchemy 使用的：
    postgresql+asyncpg://...

    LangGraph 官方 PostgresStore 底层走 psycopg，需要的是：
    postgresql://...
    """

    return dsn.replace("postgresql+asyncpg://", "postgresql://", 1)


@asynccontextmanager
async def postgres_memory_store(
    settings: Settings,
    *,
    setup: bool = True,
) -> AsyncIterator[AsyncPostgresStore]:
    """创建 LangGraph 官方 PostgresStore。

    这个 Store 用来承载长期记忆，不在这里自建 repository/service。
    第一次使用时需要执行 setup()，由官方库创建所需表结构。
    """

    conn_string = to_psycopg_conn_string(settings.postgres_dsn)
    async with AsyncPostgresStore.from_conn_string(conn_string) as store:
        if setup:
            await store.setup()
        yield store
