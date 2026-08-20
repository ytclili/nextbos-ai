from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.core.config import Settings


@asynccontextmanager
async def redis_checkpointer(settings: Settings) -> AsyncIterator[AsyncRedisSaver]:
    ttl_minutes = settings.redis_checkpoint_ttl_seconds / 60
    ttl = {"default_ttl": ttl_minutes, "refresh_on_read": True}
    async with AsyncRedisSaver.from_conn_string(settings.redis_url, ttl=ttl) as saver:
        yield saver
