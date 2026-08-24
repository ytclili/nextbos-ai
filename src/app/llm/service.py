from app.llm.models import EffectiveModelConfig
from app.llm.ports import ChatCompletion, LLMClientFactory
from app.llm.providers.factory import create_llm_client


class LLMService:
    """应用层面对 agent 暴露的 LLM 入口。

    智能体节点应该调用这个服务，而不是直接 import 具体供应商包。这样模型路由、
    测试和可观测性都能集中在同一层收口。
    """

    def __init__(self, client_factory: LLMClientFactory = create_llm_client):
        self.client_factory = client_factory

    async def chat(
        self,
        *,
        messages: list[object],
        config: EffectiveModelConfig,
    ) -> ChatCompletion:
        client = self.client_factory(config)
        return await client.chat(messages=messages, config=config)
