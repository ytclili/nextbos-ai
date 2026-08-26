from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from opentelemetry import trace
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from app.agent.options import ChatModelOptions
from app.api.errors import map_exception_to_http_error
from app.api.streaming import encode_sse_event
from app.conversation.identity import (
    ConversationActor,
    new_thread_id,
    new_visitor_id,
    resolve_conversation_actor,
)
from app.services.agent_service import AgentService

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatModelParams(BaseModel):
    """单次 chat 请求允许覆盖的大模型参数。"""

    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=None, ge=0, le=1)
    timeout_seconds: int | None = Field(default=None, ge=1)


class ChatRequest(BaseModel):
    thread_id: str | None = Field(default=None, min_length=1, max_length=128)
    user_id: str | None = Field(default=None, min_length=1, max_length=120)
    visitor_id: str | None = Field(default=None, min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=20_000)
    token: str | None = Field(default=None, min_length=1, max_length=4096)

    # 可选：选择数据库里已经登记过的模型 alias。
    model_alias: str | None = Field(default=None, min_length=1, max_length=128)

    # 可选：本次请求覆盖模型生成参数。
    model_params: ChatModelParams | None = None


class ChatResumePayload(BaseModel):
    """前端完成 HITL 动作后传回来的 resume 数据。"""

    type: Literal["auth_result"]
    status: Literal["success", "failed"]
    token: str | None = Field(default=None, min_length=1, max_length=4096)
    reason: str | None = Field(default=None, max_length=500)


class ChatResumeRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    user_id: str | None = Field(default=None, min_length=1, max_length=120)
    visitor_id: str | None = Field(default=None, min_length=1, max_length=120)
    resume: ChatResumePayload

    # resume 时通常沿用 checkpoint 里的上下文；这里保留模型覆盖入口，
    # 避免前端在特殊调试场景下无法指定模型参数。
    model_alias: str | None = Field(default=None, min_length=1, max_length=128)
    model_params: ChatModelParams | None = None


class ChatResponseItem(BaseModel):
    """返回给前端渲染的一个消息块。"""

    type: str = Field(description="前端渲染类型，例如 text、card、table、form、action")
    content: str | None = Field(default=None, description="文本内容；卡片/表格类消息可为空")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="前端渲染所需的数据，例如卡片字段、按钮、跳转参数",
    )


class ChatResponseData(BaseModel):
    """chat 接口 data 字段里的业务数据。"""

    thread_id: str
    user_id: str | None = None
    visitor_id: str | None = None
    trace_id: str | None = Field(default=None, description="本次请求的 OpenTelemetry trace_id")
    items: list[ChatResponseItem]


class ChatResponse(BaseModel):
    """chat 接口返回给前端的统一结构。"""

    code: int = 200
    status: Literal["success"] = "success"
    message: str = "success"
    data: ChatResponseData


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, http: Request) -> ChatResponse:
    trace_id = _current_trace_id()
    actor = _resolve_chat_actor(request)
    thread_id = request.thread_id or new_thread_id()

    try:
        service = AgentService(
            http.app.state.checkpointer,
            http.app.state.session_factory,
            http.app.state.settings,
            memory_store=http.app.state.memory_store,
        )
        content = await service.chat(
            thread_id=thread_id,
            user_id=actor.actor_id,
            message=request.message,
            token=request.token,
            model_options=_to_model_options(request),
            trace_id=trace_id,
        )
    except Exception as exc:
        raise map_exception_to_http_error(exc) from exc

    return ChatResponse(
        data=ChatResponseData(
            thread_id=thread_id,
            user_id=actor.user_id,
            visitor_id=actor.visitor_id,
            trace_id=trace_id,
            items=[
                ChatResponseItem(
                    type="text",
                    content=content,
                )
            ],
        )
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest, http: Request) -> StreamingResponse:
    """流式 chat 接口。

    返回 text/event-stream：
    - start：请求开始；
    - token：模型增量文本；
    - done：模型完整回复已经生成并落库；
    - error：流式过程中出现异常。
    """

    trace_id = _current_trace_id()
    actor = _resolve_chat_actor(request)
    thread_id = request.thread_id or new_thread_id()
    service = AgentService(
        http.app.state.checkpointer,
        http.app.state.session_factory,
        http.app.state.settings,
        memory_store=http.app.state.memory_store,
    )

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event, data in service.stream_chat(
                thread_id=thread_id,
                user_id=actor.actor_id,
                message=request.message,
                token=request.token,
                model_options=_to_model_options(request),
                trace_id=trace_id,
            ):
                yield encode_sse_event(event, data)
                if event == "start" and actor.generated_visitor_id and actor.visitor_id:
                    yield encode_sse_event(
                        "visitor",
                        {
                            "thread_id": thread_id,
                            "visitor_id": actor.visitor_id,
                        },
                    )
        except Exception as exc:
            http_error = map_exception_to_http_error(exc)
            detail = http_error.detail
            if isinstance(detail, dict):
                code = str(detail.get("code", "stream_error"))
                message = str(detail.get("message", "流式输出失败，请稍后重试。"))
            else:
                code = "stream_error"
                message = str(detail)
            yield encode_sse_event(
                "error",
                {
                    "code": code,
                    "message": message,
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/resume/stream")
async def chat_resume_stream(
    request: ChatResumeRequest,
    http: Request,
) -> StreamingResponse:
    """从 LangGraph interrupt 恢复流式 chat。"""

    trace_id = _current_trace_id()
    actor = _resolve_resume_actor(request)
    service = AgentService(
        http.app.state.checkpointer,
        http.app.state.session_factory,
        http.app.state.settings,
        memory_store=http.app.state.memory_store,
    )

    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event, data in service.stream_resume_chat(
                thread_id=request.thread_id,
                user_id=actor.actor_id,
                resume=request.resume.model_dump(exclude_none=True),
                model_options=_to_model_options(request),
                trace_id=trace_id,
            ):
                yield encode_sse_event(event, data)
        except Exception as exc:
            http_error = map_exception_to_http_error(exc)
            detail = http_error.detail
            if isinstance(detail, dict):
                code = str(detail.get("code", "stream_error"))
                message = str(detail.get("message", "流式输出失败，请稍后重试。"))
            else:
                code = "stream_error"
                message = str(detail)
            yield encode_sse_event(
                "error",
                {
                    "code": code,
                    "message": message,
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _resolve_chat_actor(request: ChatRequest) -> ConversationActor:
    """解析 chat 请求身份；匿名请求自动生成 visitor_id。"""

    if request.user_id is None and request.visitor_id is None:
        visitor_id = new_visitor_id()
        return ConversationActor(
            actor_id=f"visitor:{visitor_id}",
            visitor_id=visitor_id,
            generated_visitor_id=True,
        )

    return resolve_conversation_actor(
        user_id=request.user_id,
        visitor_id=request.visitor_id,
        allow_generate_visitor=False,
    )


def _resolve_resume_actor(request: ChatResumeRequest) -> ConversationActor:
    """解析 resume 请求身份；恢复 checkpoint 时不能重新生成游客身份。"""

    try:
        return resolve_conversation_actor(
            user_id=request.user_id,
            visitor_id=request.visitor_id,
            allow_generate_visitor=False,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "identity_required",
                "message": "resume 请求必须携带 user_id 或 visitor_id。",
            },
        ) from exc


def _to_model_options(request: ChatRequest | ChatResumeRequest) -> ChatModelOptions:
    """把接口请求里的模型参数转换成 agent service 使用的对象。"""

    return ChatModelOptions(
        model_alias=request.model_alias,
        model_params=request.model_params.model_dump(exclude_none=True)
        if request.model_params
        else None,
    )


def _current_trace_id() -> str | None:
    """读取当前请求所在的 trace_id。

    如果没有启用 OTel，或者当前不在有效 span 内，则返回 None。
    """

    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None
    return format(span_context.trace_id, "032x")
