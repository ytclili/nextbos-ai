from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.agent.options import ChatModelOptions
from app.services.agent_service import AgentService

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatModelParams(BaseModel):
    """单次 chat 请求允许覆盖的大模型参数。"""

    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0, le=1)
    timeout_seconds: int | None = Field(default=None, ge=1)


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    user_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=20_000)

    # 可选：选择数据库里已经登记过的模型 alias。
    model_alias: str | None = Field(default=None, min_length=1, max_length=128)

    # 可选：本次请求覆盖模型生成参数。
    model_params: ChatModelParams | None = None


class ChatResponseItem(BaseModel):
    """返回给前端渲染的一个消息块。"""

    type: str = Field(description="前端渲染类型，例如 text、card、table、form、action")
    content: str | None = Field(default=None, description="文本内容；卡片/表格类消息可为空")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="前端渲染所需的数据，例如卡片字段、按钮、跳转参数",
    )


class ChatResponse(BaseModel):
    """chat 接口返回给前端的统一结构。"""

    status: Literal["success"] = "success"
    thread_id: str
    items: list[ChatResponseItem]


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, http: Request) -> ChatResponse:
    try:
        service = AgentService(
            http.app.state.checkpointer,
            http.app.state.session_factory,
            http.app.state.settings,
        )
        content = await service.chat(
            thread_id=request.thread_id,
            user_id=request.user_id,
            message=request.message,
            model_options=ChatModelOptions(
                model_alias=request.model_alias,
                model_params=request.model_params.model_dump(exclude_none=True)
                if request.model_params
                else None,
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"agent infrastructure unavailable: {exc}",
        ) from exc

    return ChatResponse(
        thread_id=request.thread_id,
        items=[
            ChatResponseItem(
                type="text",
                content=content,
            )
        ],
    )
