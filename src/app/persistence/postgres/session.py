from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import Settings


def create_session_factory(
    settings: Settings | None = None,
    *,
    engine: AsyncEngine | None = None,
) -> async_sessionmaker[AsyncSession]:
    """创建异步数据库 session factory。

    优先复用传入的 engine，避免服务启动时创建两个连接池。
    settings 只作为兼容旧调用方式的兜底入口。
    """

    if engine is None:
        if settings is None:
            raise ValueError("settings or engine is required")
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(settings.postgres_dsn, pool_pre_ping=True)

    return async_sessionmaker(engine, expire_on_commit=False)
