from typing import Protocol


class KnowledgeRetriever(Protocol):
    async def search(self, query: str, *, limit: int = 5) -> list[dict[str, object]]: ...
