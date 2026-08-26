from dataclasses import dataclass
from uuid import uuid4

ACTOR_USER_PREFIX = "user:"
ACTOR_VISITOR_PREFIX = "visitor:"


@dataclass(frozen=True)
class ConversationActor:
    """会话归属身份。

    对外接口区分 user_id / visitor_id；
    内部存储和 LangGraph namespace 统一使用 actor_id。
    """

    actor_id: str
    user_id: str | None = None
    visitor_id: str | None = None
    generated_visitor_id: bool = False


def new_thread_id() -> str:
    """生成对外暴露的会话 ID。"""

    return str(uuid4())


def new_visitor_id() -> str:
    """生成游客临时身份 ID。"""

    return f"visitor_{uuid4().hex}"


def resolve_conversation_actor(
    *,
    user_id: str | None = None,
    visitor_id: str | None = None,
    allow_generate_visitor: bool = False,
) -> ConversationActor:
    """把接口身份参数解析成内部 actor_id。

    user_id 优先级高于 visitor_id；两者都没有时，按需生成游客身份。
    """

    normalized_user_id = _clean_identity(user_id)
    if normalized_user_id:
        return ConversationActor(
            actor_id=f"{ACTOR_USER_PREFIX}{normalized_user_id}",
            user_id=normalized_user_id,
        )

    normalized_visitor_id = _clean_identity(visitor_id)
    if normalized_visitor_id:
        return ConversationActor(
            actor_id=f"{ACTOR_VISITOR_PREFIX}{normalized_visitor_id}",
            visitor_id=normalized_visitor_id,
        )

    if allow_generate_visitor:
        generated_visitor_id = new_visitor_id()
        return ConversationActor(
            actor_id=f"{ACTOR_VISITOR_PREFIX}{generated_visitor_id}",
            visitor_id=generated_visitor_id,
            generated_visitor_id=True,
        )

    raise ValueError("user_id or visitor_id is required")


def split_actor_id(actor_id: str) -> ConversationActor:
    """把内部 actor_id 反解成接口身份字段。"""

    if actor_id.startswith(ACTOR_USER_PREFIX):
        user_id = actor_id.removeprefix(ACTOR_USER_PREFIX)
        return ConversationActor(actor_id=actor_id, user_id=user_id)

    if actor_id.startswith(ACTOR_VISITOR_PREFIX):
        visitor_id = actor_id.removeprefix(ACTOR_VISITOR_PREFIX)
        return ConversationActor(actor_id=actor_id, visitor_id=visitor_id)

    # 兼容旧数据：历史记录里直接存真实 user_id。
    return ConversationActor(actor_id=actor_id, user_id=actor_id)


def _clean_identity(value: str | None) -> str | None:
    """清理接口传入的身份字段。"""

    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None
