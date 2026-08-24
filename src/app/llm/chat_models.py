from collections.abc import Callable
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from app.llm.errors import LLMConfigurationError
from app.llm.models import EffectiveModelConfig


def create_langchain_chat_model(
    config: EffectiveModelConfig,
    *,
    chat_model_factory: Callable[..., Any] | None = None,
) -> Any:
    """根据最终模型配置创建 LangChain 聊天模型。

    这里是项目和 LangChain 的边界：
    - 项目负责从 DB/env 解析 EffectiveModelConfig；
    - LangChain 负责返回标准 ChatModel、AIMessage、ToolMessage、tool_calls。

    agent 图里应该直接使用这个模型对象，不再把 AIMessage 包成自定义 DTO。
    """

    factory = chat_model_factory or _default_chat_model_factory
    return factory(**build_chat_model_kwargs(config))


def normalize_langchain_messages(messages: list[object]) -> list[BaseMessage]:
    """把普通字符串等输入规整成 LangChain 消息对象。"""

    return [
        message if isinstance(message, BaseMessage) else HumanMessage(content=str(message))
        for message in messages
    ]


def build_chat_model_kwargs(config: EffectiveModelConfig) -> dict[str, object]:
    """把 EffectiveModelConfig 转成 LangChain ChatOpenAI 参数。"""

    if not config.base_url or not config.model_name:
        raise LLMConfigurationError("LLM base_url and model_name are required")
    if config.credential is None or not config.credential.api_key:
        raise LLMConfigurationError("LLM api key is required")

    # ChatOpenAI 支持 OpenAI-compatible base_url，因此 DeepSeek、Qwen、Kimi、
    # OpenAI 等兼容接口都可以先复用这一层。
    kwargs: dict[str, object] = {
        "model": config.model_name,
        "base_url": config.base_url,
        "api_key": config.credential.api_key,
        "timeout": config.params.get("timeout_seconds", 60),
    }
    for key in ("temperature", "max_tokens", "top_p"):
        value = config.params.get(key)
        if value is not None:
            kwargs[key] = value
    return kwargs


def _default_chat_model_factory(**kwargs):
    """延迟导入 LangChain 集成包，避免测试假模型时强依赖真实供应商。"""

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise LLMConfigurationError(
            "langchain-openai is required for openai-compatible chat models"
        ) from exc
    return ChatOpenAI(**kwargs)
