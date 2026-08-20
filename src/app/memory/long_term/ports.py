from typing import Protocol

from app.memory.long_term.models import Memory


class LongTermMemoryStore(Protocol):
    async def search(
        self,
        *,
        namespace: tuple[str, ...],
        query: str,
        limit: int = 10,
    ) -> list[Memory]: ...
    async def upsert(self, memory: Memory) -> Memory: ...
    async def delete(self, memory_id: str) -> None: ...
