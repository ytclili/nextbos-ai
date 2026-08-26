from contextvars import ContextVar, Token

from app.core.config import Settings

_core_internal_token: ContextVar[str | None] = ContextVar(
    "core_internal_token",
    default=None,
)


class CoreInternalTokenOverride:
    """本次工具调用的核心业务系统 token override。"""

    def __init__(self, token: Token[str | None]) -> None:
        self._token = token

    def reset(self) -> None:
        """恢复进入 override 前的 token。"""

        _core_internal_token.reset(self._token)


def use_core_internal_token(token: str | None) -> CoreInternalTokenOverride:
    """设置本次上下文里的核心业务系统 token。"""

    normalized = token.strip() if isinstance(token, str) and token.strip() else None
    return CoreInternalTokenOverride(_core_internal_token.set(normalized))


def core_internal_headers(settings: Settings) -> dict[str, str]:
    """生成调用核心业务系统需要的鉴权 header。"""

    token = _core_internal_token.get() or settings.core_internal_token
    if not token:
        return {}

    normalized = token.strip()
    return {
        "Authorization": (
            normalized if normalized.lower().startswith("bearer ") else f"Bearer {normalized}"
        )
    }
