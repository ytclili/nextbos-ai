from app.llm.models import EffectiveModelConfig
from app.llm.ports import LLMClient, LLMConfigurationError
from app.llm.providers.langchain_chat import LangChainChatClient


def create_llm_client(config: EffectiveModelConfig) -> LLMClient:
    """根据解析后的供应商类型创建对应的 LangChain 适配器。"""

    if config.provider == "openai_compatible":
        return LangChainChatClient()
    raise LLMConfigurationError(f"Unsupported LLM provider: {config.provider}")
