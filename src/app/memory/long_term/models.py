from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class Memory(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID = Field(default_factory=uuid4)
    namespace: tuple[str, ...]
    memory_type: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
    importance: float = 0.5
    confidence: float = 1.0
    source_thread_id: str | None = None
    content_hash: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
