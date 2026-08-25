from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langmem.short_term import SummarizationNode

from app.agent.nodes.respond import create_respond_node
from app.agent.nodes.summarize import (
    SummaryOptions,
    create_summarize_node,
    skip_summarization,
)
from app.llm.models import EffectiveModelConfig, ProviderCredential


class FakeAgentModelRuntime:
    """测试用模型运行时。

    这里不连接数据库，也不调用真实大模型。
    它只记录 respond 节点传进来的参数，然后返回一个固定的假回复。
    """

    def __init__(self) -> None:
        self.resolve_calls: list[object | None] = []
        self.created_configs: list[EffectiveModelConfig] = []
        self.chat_model: FakeChatModel | None = None
        self.snapshot_id = uuid4()

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
            snapshot_id=self.snapshot_id,
        )

    def create_chat_model(self, config: EffectiveModelConfig):
        """模拟创建 LangChain ChatModel。"""

        self.created_configs.append(config)
        self.chat_model = FakeChatModel()
        return self.chat_model


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
    assert result["llm_snapshot_id"] == model_runtime.snapshot_id


@pytest.mark.asyncio
async def test_respond_node_prefers_summarized_messages_for_llm_input() -> None:
    """respond 节点应该优先把 summary 节点产出的消息传给大模型。"""

    model_runtime = FakeAgentModelRuntime()
    node = create_respond_node(model_runtime=model_runtime)
    raw_messages = [HumanMessage(content="原始旧消息")]
    summarized_messages = [HumanMessage(content="摘要后的消息")]

    await node(
        {
            "messages": raw_messages,
            "summarized_messages": summarized_messages,
            "model_options": None,
        }
    )

    assert model_runtime.chat_model is not None
    model_messages = model_runtime.chat_model.messages
    assert isinstance(model_messages[0], SystemMessage)
    assert model_messages[1:] == summarized_messages


@pytest.mark.asyncio
async def test_respond_node_prepends_system_prompt_without_persisting_it() -> None:
    """respond 节点应该把系统提示词放到模型输入首位，但不写回 graph state。"""

    model_runtime = FakeAgentModelRuntime()
    node = create_respond_node(model_runtime=model_runtime)
    raw_messages = [HumanMessage(content="今天回款怎么样？")]

    result = await node(
        {
            "messages": raw_messages,
            "model_options": None,
        }
    )

    assert model_runtime.chat_model is not None
    model_messages = model_runtime.chat_model.messages
    assert isinstance(model_messages[0], SystemMessage)
    assert "收单吧" in model_messages[0].content
    assert model_messages[1:] == raw_messages
    assert result["messages"] == [AIMessage(content="假模型回复")]


def test_summarize_node_skips_when_model_is_missing() -> None:
    """没有 summary 模型时，summarize 节点应该把 messages 原样传给 respond。"""

    messages = [HumanMessage(content="今天吃什么？")]

    assert skip_summarization({"messages": messages}) == {
        "summarized_messages": messages,
    }
    assert create_summarize_node(None) is skip_summarization


def test_summarize_node_uses_langmem_summarization_node_when_model_exists() -> None:
    """有 summary 模型时，应该使用 LangMem 官方 SummarizationNode。"""

    summary_model = object()
    node = create_summarize_node(
        summary_model,
        options=SummaryOptions(
            max_tokens=1200,
            trigger_tokens=900,
            max_output_tokens=300,
        ),
    )

    assert isinstance(node, SummarizationNode)
    assert node.max_tokens == 1200
    assert node.max_tokens_before_summary == 900
    assert node.max_summary_tokens == 300
