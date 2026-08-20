from app.knowledge.ports import KnowledgeRetriever


class KnowledgeService:
    def __init__(self, retriever: KnowledgeRetriever):
        self.retriever = retriever

    async def search(self, query: str, *, limit: int = 5) -> list[dict[str, object]]:
        return await self.retriever.search(query, limit=limit)
