from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.memory.long_term.models import Memory
from app.persistence.postgres.models import MemoryRecord


class PostgresMemoryRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def search(
        self,
        *,
        namespace: tuple[str, ...],
        query: str,
        limit: int = 10,
    ) -> list[Memory]:
        statement = select(MemoryRecord).where(
            MemoryRecord.namespace == list(namespace), MemoryRecord.content.ilike(f"%{query}%")
        ).limit(limit)
        rows = (await self.session.scalars(statement)).all()
        return [self._to_domain(row) for row in rows]

    async def upsert(self, memory: Memory) -> Memory:
        row = await self.session.get(MemoryRecord, memory.id)
        if row is None:
            row = MemoryRecord(id=memory.id)
            self.session.add(row)
        row.namespace = list(memory.namespace)
        row.memory_type = memory.memory_type
        row.content = memory.content
        row.metadata_json = memory.metadata
        row.importance = memory.importance
        row.confidence = memory.confidence
        row.source_thread_id = memory.source_thread_id
        row.content_hash = memory.content_hash
        await self.session.commit()
        await self.session.refresh(row)
        return self._to_domain(row)

    async def delete(self, memory_id: str) -> None:
        await self.session.execute(delete(MemoryRecord).where(MemoryRecord.id == UUID(memory_id)))
        await self.session.commit()

    @staticmethod
    def _to_domain(row: MemoryRecord) -> Memory:
        return Memory(
            id=row.id, namespace=tuple(row.namespace), memory_type=row.memory_type,
            content=row.content, metadata=row.metadata_json, importance=row.importance,
            confidence=row.confidence, source_thread_id=row.source_thread_id,
            content_hash=row.content_hash, created_at=row.created_at, updated_at=row.updated_at,
        )
