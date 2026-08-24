from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.llm.models import EffectiveModelConfig
from app.llm.ports import ChatCompletion, LLMConfigurationError, LLMError


class LangChainChatClient:
    """基于 LangChain 聊天模型集成的 LLM 客户端。

    这里不直接拼 OpenAI HTTP 请求，而是统一走 LangChain 的聊天模型接口。
    后续 tool calling、streaming、callback/trace 都可以继续贴着 LangChain/LangGraph 生态扩展。
    """

    def __init__(self, chat_model_factory: Callable[..., Any] | None = None):
        self.chat_model_factory = chat_model_factory or self._default_chat_model_factory

    async def chat(
        self,
        *,
        messages: list[object],
        config: EffectiveModelConfig,
    ) -> ChatCompletion:
        # 每次调用都从 EffectiveModelConfig 构造聊天模型，避免供应商客户端
        # 自己长期持有密钥或实时配置。
        chat_model = self.chat_model_factory(**self._chat_model_kwargs(config))

        try:
            response = await chat_model.ainvoke(self._to_langchain_messages(messages))
        except Exception as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        if not isinstance(response, AIMessage):
            raise LLMError("LangChain chat model returned a non-AIMessage response")

        return ChatCompletion(
            content=str(response.content),
            usage=dict(response.usage_metadata or {}),
            raw=dict(response.response_metadata or {}),
        )

    @staticmethod
    def _chat_model_kwargs(config: EffectiveModelConfig) -> dict[str, object]:
        if not config.base_url or not config.model_name:
            raise LLMConfigurationError("LLM base_url and model_name are required")
        if config.credential is None or not config.credential.api_key:
            raise LLMConfigurationError("LLM api key is required")

        # LangChain 的 ChatOpenAI 支持 OpenAI-compatible base_url，因此 DeepSeek、
        # Qwen、Kimi、OpenAI 等兼容接口都可以先复用这一层。
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

    @staticmethod
    def _to_langchain_messages(messages: list[object]) -> list[BaseMessage]:
        """把普通字符串等输入规整成 LangChain 消息对象。"""

        return [
            message if isinstance(message, BaseMessage) else HumanMessage(content=str(message))
            for message in messages
        ]

    @staticmethod
    def _default_chat_model_factory(**kwargs):
        """延迟导入 LangChain 集成包，避免测试假模型时强依赖真实供应商。"""

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise LLMConfigurationError(
                "langchain-openai is required for openai-compatible chat models"
            ) from exc
        return ChatOpenAI(**kwargs)
