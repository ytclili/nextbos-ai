from app.core.tracing import get_tracer, set_span_attributes
from app.llm.models import EffectiveModelConfig
from app.llm.ports import ChatCompletion, LLMClientFactory
from app.llm.providers.factory import create_llm_client

tracer = get_tracer(__name__)


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
        with tracer.start_as_current_span("llm.provider.chat") as span:
            span.set_attribute("llm.provider", config.provider)
            span.set_attribute("llm.model_name", config.model_name)
            span.set_attribute("llm.config.source", config.source)
            span.set_attribute("llm.config.digest", config.digest)
            span.set_attribute("llm.messages.count", len(messages))

            client = self.client_factory(config)
            completion = await client.chat(messages=messages, config=config)

            span.set_attribute("llm.response.content_length", len(completion.content))
            set_span_attributes(span, "llm.usage", completion.usage)

            return completion
