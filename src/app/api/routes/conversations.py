from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.api.errors import map_exception_to_http_error
from app.conversation.identity import (
    ConversationActor,
    resolve_conversation_actor,
    split_actor_id,
)
from app.conversation.repository import ConversationRepository
from app.persistence.postgres.models import ConversationMessage, ConversationThread

router = APIRouter(prefix="/conversations", tags=["conversations"])


class CreateConversationRequest(BaseModel):
    """创建新会话请求。"""

    user_id: str | None = Field(default=None, min_length=1, max_length=120)
    visitor_id: str | None = Field(default=None, min_length=1, max_length=120)
    title: str | None = Field(default=None, min_length=1, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationResponse(BaseModel):
    """会话线程响应。"""

    thread_id: str
    user_id: str | None = None
    visitor_id: str | None = None
    title: str | None
    status: str
    message_count: int
    last_message_at: datetime | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ConversationCreateResponse(BaseModel):
    """创建会话接口统一响应。"""

    code: int = 200
    status: str = "success"
    message: str = "success"
    data: ConversationResponse


class ConversationListResponse(BaseModel):
    """会话列表统一响应。"""

    code: int = 200
    status: str = "success"
    message: str = "success"
    data: list[ConversationResponse]
    total: int
    limit: int


class ConversationMessageResponse(BaseModel):
    """会话历史消息响应。"""

    message_id: str
    thread_id: str
    user_id: str | None = None
    visitor_id: str | None = None
    role: str
    type: str
    content: str | None
    metadata: dict[str, Any]
    status: str
    created_at: datetime


class ConversationMessageListResponse(BaseModel):
    """会话历史统一响应。"""

    code: int = 200
    status: str = "success"
    message: str = "success"
    data: list[ConversationMessageResponse]
    total: int
    limit: int


class BindVisitorRequest(BaseModel):
    """登录后把游客会话归属绑定到真实用户。"""

    visitor_id: str = Field(min_length=1, max_length=120)
    user_id: str = Field(min_length=1, max_length=120)


class BindVisitorData(BaseModel):
    visitor_id: str
    user_id: str
    updated_threads: int
    updated_messages: int
    updated_summaries: int


class BindVisitorResponse(BaseModel):
    """绑定游客会话接口统一响应。"""

    code: int = 200
    status: str = "success"
    message: str = "success"
    data: BindVisitorData


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    http: Request,
    user_id: str | None = Query(default=None, min_length=1, max_length=120),
    visitor_id: str | None = Query(default=None, min_length=1, max_length=120),
    limit: int = Query(default=20, ge=1, le=100),
) -> ConversationListResponse:
    """获取某个用户的会话列表。

    这里只读取 conversation_threads：
    - 不读取 Redis checkpoint；
    - 不读取 messages 明细；
    - 不调用 LangGraph / LLM。
    """

    actor = _resolve_actor_or_422(user_id=user_id, visitor_id=visitor_id)
    try:
        repository = ConversationRepository(http.app.state.session_factory)
        threads = await repository.list_threads(user_id=actor.actor_id, limit=limit)
        total = await repository.count_threads(user_id=actor.actor_id)
    except Exception as exc:
        raise map_exception_to_http_error(exc) from exc

    return ConversationListResponse(
        data=[_thread_to_response(thread) for thread in threads],
        total=total,
        limit=limit,
    )


@router.get("/{thread_id}/messages", response_model=ConversationMessageListResponse)
async def list_conversation_messages(
    thread_id: str,
    http: Request,
    user_id: str | None = Query(default=None, min_length=1, max_length=120),
    visitor_id: str | None = Query(default=None, min_length=1, max_length=120),
    limit: int = Query(default=50, ge=1, le=200),
) -> ConversationMessageListResponse:
    """获取某个会话的历史消息。

    这里从 PostgreSQL 的 conversation_messages 读取：
    - 不读取 Redis checkpoint；
    - 不反解 LangGraph 内部状态；
    - 同时使用 thread_id 和 user_id 过滤，避免越权读取。
    """

    actor = _resolve_actor_or_422(user_id=user_id, visitor_id=visitor_id)
    try:
        repository = ConversationRepository(http.app.state.session_factory)
        messages = await repository.list_messages(
            thread_id=thread_id,
            user_id=actor.actor_id,
            limit=limit,
        )
        total = await repository.count_messages(thread_id=thread_id, user_id=actor.actor_id)
    except Exception as exc:
        raise map_exception_to_http_error(exc) from exc

    return ConversationMessageListResponse(
        data=[_message_to_response(message) for message in messages],
        total=total,
        limit=limit,
    )


@router.post("", response_model=ConversationCreateResponse)
async def create_conversation(
    request: CreateConversationRequest,
    http: Request,
) -> ConversationCreateResponse:
    """创建一个空会话。

    这里只创建 conversation_threads：
    - 不写 conversation_messages；
    - 不创建 Redis checkpoint；
    - 不调用 LangGraph / LLM。
    """

    try:
        actor = resolve_conversation_actor(
            user_id=request.user_id,
            visitor_id=request.visitor_id,
            allow_generate_visitor=True,
        )
        repository = ConversationRepository(http.app.state.session_factory)
        thread = await repository.create_thread(
            user_id=actor.actor_id,
            title=request.title,
            metadata=request.metadata,
        )
    except Exception as exc:
        raise map_exception_to_http_error(exc) from exc

    return ConversationCreateResponse(data=_thread_to_response(thread))


@router.post("/bind-visitor", response_model=BindVisitorResponse)
async def bind_visitor_conversations(
    request: BindVisitorRequest,
    http: Request,
) -> BindVisitorResponse:
    """登录成功后，把游客会话归属迁移到真实用户。"""

    visitor_actor = resolve_conversation_actor(visitor_id=request.visitor_id)
    user_actor = resolve_conversation_actor(user_id=request.user_id)

    try:
        repository = ConversationRepository(http.app.state.session_factory)
        result = await repository.bind_actor(
            from_actor_id=visitor_actor.actor_id,
            to_actor_id=user_actor.actor_id,
        )
    except Exception as exc:
        raise map_exception_to_http_error(exc) from exc

    return BindVisitorResponse(
        data=BindVisitorData(
            visitor_id=request.visitor_id,
            user_id=request.user_id,
            updated_threads=int(result.get("updated_threads", 0)),
            updated_messages=int(result.get("updated_messages", 0)),
            updated_summaries=int(result.get("updated_summaries", 0)),
        )
    )


def _thread_to_response(thread: ConversationThread) -> ConversationResponse:
    """把 ORM thread 行转换成接口响应。"""

    actor = split_actor_id(thread.user_id)
    return ConversationResponse(
        thread_id=thread.thread_id,
        user_id=actor.user_id,
        visitor_id=actor.visitor_id,
        title=thread.title,
        status=thread.status,
        message_count=thread.message_count,
        last_message_at=thread.last_message_at,
        metadata=thread.metadata_json or {},
        created_at=thread.created_at,
        updated_at=thread.updated_at,
    )


def _message_to_response(message: ConversationMessage) -> ConversationMessageResponse:
    """把 ORM message 行转换成接口响应。"""

    actor = split_actor_id(message.user_id)
    return ConversationMessageResponse(
        message_id=str(message.id),
        thread_id=message.thread_id,
        user_id=actor.user_id,
        visitor_id=actor.visitor_id,
        role=message.role,
        type=message.type,
        content=message.content,
        metadata=message.metadata_json or {},
        status=message.status,
        created_at=message.created_at,
    )


def _resolve_actor_or_422(
    *,
    user_id: str | None,
    visitor_id: str | None,
) -> ConversationActor:
    """解析查询接口身份；列表和历史接口不能静默生成游客身份。"""

    try:
        return resolve_conversation_actor(
            user_id=user_id,
            visitor_id=visitor_id,
            allow_generate_visitor=False,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "identity_required",
                "message": "请求必须携带 user_id 或 visitor_id。",
            },
        ) from exc
