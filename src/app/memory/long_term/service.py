import hashlib

from app.memory.long_term.models import Memory
from app.memory.long_term.ports import LongTermMemoryStore


class LongTermMemoryService:
    def __init__(self, store: LongTermMemoryStore):
        self.store = store

    async def remember(self, *, namespace: tuple[str, ...], memory_type: str, content: str,
                       source_thread_id: str | None = None,
                       metadata: dict[str, object] | None = None) -> Memory:
        memory = Memory(
            namespace=namespace,
            memory_type=memory_type,
            content=content,
            metadata=metadata or {},
            source_thread_id=source_thread_id,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
        return await self.store.upsert(memory)
