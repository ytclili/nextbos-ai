from contextlib import asynccontextmanager

import pytest

from app.core.config import Settings
from app.memory.long_term import store as store_module
from app.memory.long_term.store import postgres_memory_store, to_psycopg_conn_string


def test_to_psycopg_conn_string_converts_sqlalchemy_asyncpg_dsn() -> None:
    """LangGraph PostgresStore 应该收到 psycopg 可识别的 DSN。"""

    assert (
        to_psycopg_conn_string("postgresql+asyncpg://agent:agent@localhost:5432/db")
        == "postgresql://agent:agent@localhost:5432/db"
    )


def test_to_psycopg_conn_string_keeps_plain_postgres_dsn() -> None:
    """如果配置本来就是 psycopg DSN，就不应该改变。"""

    assert (
        to_psycopg_conn_string("postgresql://agent:agent@localhost:5432/db")
        == "postgresql://agent:agent@localhost:5432/db"
    )


@pytest.mark.asyncio
async def test_postgres_memory_store_uses_official_store_and_runs_setup(
    monkeypatch,
) -> None:
    """长期记忆 Store 应该使用 LangGraph 官方 AsyncPostgresStore。"""

    captured = {}

    class FakeStore:
        async def setup(self) -> None:
            captured["setup_called"] = True

    class FakeAsyncPostgresStore:
        @staticmethod
        @asynccontextmanager
        async def from_conn_string(conn_string):
            captured["conn_string"] = conn_string
            yield FakeStore()

    monkeypatch.setattr(store_module, "AsyncPostgresStore", FakeAsyncPostgresStore)

    settings = Settings(postgres_dsn="postgresql+asyncpg://agent:agent@localhost:5432/db")

    async with postgres_memory_store(settings) as store:
        assert isinstance(store, FakeStore)

    assert captured == {
        "conn_string": "postgresql://agent:agent@localhost:5432/db",
        "setup_called": True,
    }
