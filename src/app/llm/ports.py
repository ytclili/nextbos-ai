from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from app.llm.models import EffectiveModelConfig


class LLMError(Exception):
    """LLM 调用或响应解析失败。"""


class LLMConfigurationError(LLMError):
    """LLM 配置缺失或不受支持。"""


@dataclass(frozen=True)
class ChatCompletion:
    """所有供应商适配器返回的统一聊天结果。"""

    content: str
    usage: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] | None = None


class LLMClient(Protocol):
    """LLMService 依赖的供应商适配器接口。"""

    async def chat(
        self,
        *,
        messages: list[object],
        config: EffectiveModelConfig,
    ) -> ChatCompletion: ...


LLMClientFactory = Callable[[EffectiveModelConfig], LLMClient]
