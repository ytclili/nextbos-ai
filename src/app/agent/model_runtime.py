from typing import Any

from app.agent.options import ChatModelOptions
from app.core.tracing import get_tracer
from app.llm.chat_models import create_langchain_chat_model
from app.llm.config_resolver import ModelConfigResolver, ModelSelection, RequestedModelParams
from app.llm.models import EffectiveModelConfig

tracer = get_tracer(__name__)


class AgentModelRuntime:
    """agent 图使用的模型运行时。

    这里只负责两件事：
    1. 把请求侧 ChatModelOptions 解析成最终模型配置；
    2. 根据最终配置创建 LangChain 聊天模型。

    真正的 AIMessage、tool_calls、ToolMessage 等对象继续走 LangChain/LangGraph
    官方标准，不在这里重新包装。
    """

    def __init__(
        self,
        *,
        config_resolver: ModelConfigResolver,
        chat_model_factory=create_langchain_chat_model,
    ):
        self.config_resolver = config_resolver
        self.chat_model_factory = chat_model_factory

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

    def create_chat_model(self, config: EffectiveModelConfig):
        """创建 LangChain ChatModel。

        这个方法返回原生 LangChain 对象，respond 节点可以继续调用 bind_tools()、
        ainvoke()，不会丢失 tool_calls。
        """

        return self.chat_model_factory(config)

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
