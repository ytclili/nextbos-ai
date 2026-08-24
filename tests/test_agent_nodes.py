import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes.respond import create_respond_node
from app.llm.models import EffectiveModelConfig, ProviderCredential


class FakeAgentModelRuntime:
    """测试用模型运行时。

    这里不连接数据库，也不调用真实大模型。
    它只记录 respond 节点传进来的参数，然后返回一个固定的假回复。
    """

    def __init__(self) -> None:
        self.resolve_calls: list[object | None] = []
        self.created_configs: list[EffectiveModelConfig] = []

    async def resolve_config(self, options: object | None) -> EffectiveModelConfig:
        """模拟模型配置解析。"""

        self.resolve_calls.append(options)
        return EffectiveModelConfig(
            source="env_fallback",
            provider="openai_compatible",
            base_url="https://api.example.com/v1",
            model_name="test-model",
            params={},
            credential=ProviderCredential(
                id=None,
                provider="openai_compatible",
                name="env",
                api_key="test-key",
            ),
            digest="digest",
        )

    def create_chat_model(self, config: EffectiveModelConfig):
        """模拟创建 LangChain ChatModel。"""

        self.created_configs.append(config)
        return FakeChatModel()


class FakeChatModel:
    """测试用 LangChain ChatModel。"""

    def __init__(self):
        self.bound_tools = []
        self.messages = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return AIMessage(content="假模型回复")


@pytest.mark.asyncio
async def test_respond_node_calls_model_runtime_and_returns_ai_message() -> None:
    """respond 节点应该调用模型运行时，并把模型结果转成 AIMessage。"""

    model_runtime = FakeAgentModelRuntime()
    node = create_respond_node(model_runtime=model_runtime)

    result = await node(
        {
            "messages": [HumanMessage(content="今天吃什么？")],
            "model_options": None,
        }
    )

    assert model_runtime.resolve_calls == [None]
    assert len(model_runtime.created_configs) == 1

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "假模型回复"
