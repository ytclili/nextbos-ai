from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from app.api.errors import map_exception_to_http_error
from app.conversation.repository import ConversationRepository
from app.persistence.postgres.models import ConversationMessage, ConversationThread

router = APIRouter(prefix="/conversations", tags=["conversations"])


class CreateConversationRequest(BaseModel):
    """创建新会话请求。"""

    user_id: str = Field(min_length=1, max_length=128)
    title: str | None = Field(default=None, min_length=1, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationResponse(BaseModel):
    """会话线程响应。"""

    thread_id: str
    user_id: str
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
    user_id: str
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


@router.get("", response_model=ConversationListResponse)
async def list_conversations(
    http: Request,
    user_id: str = Query(min_length=1, max_length=128),
    limit: int = Query(default=20, ge=1, le=100),
) -> ConversationListResponse:
    """获取某个用户的会话列表。

    这里只读取 conversation_threads：
    - 不读取 Redis checkpoint；
    - 不读取 messages 明细；
    - 不调用 LangGraph / LLM。
    """

    try:
        repository = ConversationRepository(http.app.state.session_factory)
        threads = await repository.list_threads(user_id=user_id, limit=limit)
        total = await repository.count_threads(user_id=user_id)
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
    user_id: str = Query(min_length=1, max_length=128),
    limit: int = Query(default=50, ge=1, le=200),
) -> ConversationMessageListResponse:
    """获取某个会话的历史消息。

    这里从 PostgreSQL 的 conversation_messages 读取：
    - 不读取 Redis checkpoint；
    - 不反解 LangGraph 内部状态；
    - 同时使用 thread_id 和 user_id 过滤，避免越权读取。
    """

    try:
        repository = ConversationRepository(http.app.state.session_factory)
        messages = await repository.list_messages(
            thread_id=thread_id,
            user_id=user_id,
            limit=limit,
        )
        total = await repository.count_messages(thread_id=thread_id, user_id=user_id)
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
        repository = ConversationRepository(http.app.state.session_factory)
        thread = await repository.create_thread(
            user_id=request.user_id,
            title=request.title,
            metadata=request.metadata,
        )
    except Exception as exc:
        raise map_exception_to_http_error(exc) from exc

    return ConversationCreateResponse(data=_thread_to_response(thread))


def _thread_to_response(thread: ConversationThread) -> ConversationResponse:
    """把 ORM thread 行转换成接口响应。"""

    return ConversationResponse(
        thread_id=thread.thread_id,
        user_id=thread.user_id,
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

    return ConversationMessageResponse(
        message_id=str(message.id),
        thread_id=message.thread_id,
        user_id=message.user_id,
        role=message.role,
        type=message.type,
        content=message.content,
        metadata=message.metadata_json or {},
        status=message.status,
        created_at=message.created_at,
    )
