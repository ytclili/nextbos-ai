from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.agent_service import AgentService

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=20_000)


class ChatResponse(BaseModel):
    thread_id: str
    content: str


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, http: Request) -> ChatResponse:
    try:
        service = AgentService(http.app.state.checkpointer)
        content = await service.chat(
            thread_id=request.thread_id, user_id=request.user_id, message=request.message
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"agent infrastructure unavailable: {exc}",
        ) from exc
    return ChatResponse(thread_id=request.thread_id, content=content)
