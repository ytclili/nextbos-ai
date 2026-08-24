from typing import Any

from app.agent.options import ChatModelOptions
from app.core.tracing import get_tracer, set_span_attributes
from app.llm.config_resolver import ModelConfigResolver, ModelSelection, RequestedModelParams
from app.llm.models import EffectiveModelConfig
from app.llm.ports import ChatCompletion
from app.llm.service import LLMService

tracer = get_tracer(__name__)


class AgentModelRuntime:
    """agent 调用模型时使用的运行时门面。

    这里负责把接口层传入的 ChatModelOptions 转成模型配置解析器认识的对象，
    并调用 LLMService。LangGraph 节点只依赖这个门面，不直接关心配置如何解析。
    """

    def __init__(
        self,
        *,
        config_resolver: ModelConfigResolver,
        llm_service: LLMService,
    ):
        self.config_resolver = config_resolver
        self.llm_service = llm_service

    async def resolve_config(self, options: ChatModelOptions | None) -> EffectiveModelConfig:
        """根据本次请求的模型选择意图解析最终模型配置。"""

        with tracer.start_as_current_span("llm.resolve_config") as span:
            model_alias = options.model_alias if options and options.model_alias else ""
            span.set_attribute("llm.request.model_alias", model_alias)

            config = await self.config_resolver.resolve(
                selection=ModelSelection(
                    model_alias=options.model_alias if options else None,
                ),
                requested_params=self._to_requested_params(
                    options.model_params if options else None,
                ),
            )

            span.set_attribute("llm.config.source", config.source)
            span.set_attribute("llm.config.provider", config.provider)
            span.set_attribute("llm.config.model_name", config.model_name)
            span.set_attribute("llm.config.digest", config.digest)
            return config

    async def chat(
        self,
        *,
        messages: list[object],
        options: ChatModelOptions | None,
    ) -> ChatCompletion:
        """解析模型配置并调用 LLMService。"""

        config = await self.resolve_config(options)

        with tracer.start_as_current_span("llm.chat") as span:
            span.set_attribute("llm.config.source", config.source)
            span.set_attribute("llm.provider", config.provider)
            span.set_attribute("llm.model_name", config.model_name)
            span.set_attribute("llm.config.digest", config.digest)
            span.set_attribute("llm.messages.count", len(messages))

            completion = await self.llm_service.chat(messages=messages, config=config)

            span.set_attribute("llm.response.content_length", len(completion.content))
            set_span_attributes(span, "llm.usage", completion.usage)

            return completion

    @staticmethod
    def _to_requested_params(params: dict[str, Any] | None) -> RequestedModelParams:
        """把接口层传来的 dict 转成模型配置解析器认识的参数对象。"""

        params = params or {}
        return RequestedModelParams(
            temperature=params.get("temperature"),
            max_tokens=params.get("max_tokens"),
            top_p=params.get("top_p"),
            timeout_seconds=params.get("timeout_seconds"),
        )
