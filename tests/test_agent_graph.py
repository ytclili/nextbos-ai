import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent.graph import build_graph
from app.llm.models import EffectiveModelConfig, ProviderCredential


class FakeAgentModelRuntime:
    """测试用模型运行时。

    这个假对象用来替代真实 AgentModelRuntime。
    这样测试 graph 的时候不会访问数据库，也不会请求真实大模型。
    """

    def __init__(self) -> None:
        self.resolve_calls: list[object | None] = []
        self.chat_model = FakeChatModel()

    async def resolve_config(self, options: object | None) -> EffectiveModelConfig:
        """模拟配置解析，并记录 graph 传进来的参数。"""

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
        return self.chat_model


class FakeChatModel:
    """测试用 LangChain ChatModel。"""

    def __init__(self) -> None:
        self.calls: list[list[object]] = []

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="今天可以吃牛肉面")


class ToolCallingAgentModelRuntime(FakeAgentModelRuntime):
    """先请求调用工具，再基于工具结果返回最终回复。"""

    def __init__(self) -> None:
        super().__init__()
        self.chat_model = ToolCallingChatModel()


class ToolCallingChatModel(FakeChatModel):
    """模拟支持 tool calling 的 LangChain ChatModel。"""

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "health_check",
                        "args": {},
                        "id": "call-health-check",
                    }
                ],
            )
        return AIMessage(content="工具系统正常")


@pytest.mark.asyncio
async def test_graph_runs_start_to_respond_to_end_with_model_runtime() -> None:
    """graph 应该执行 respond 节点，并把模型回复追加到 messages。"""

    model_runtime = FakeAgentModelRuntime()
    graph = build_graph(model_runtime=model_runtime)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="今天吃什么？")],
            "model_options": None,
        }
    )

    assert model_runtime.resolve_calls == [None]

    called_messages = model_runtime.chat_model.calls[0]
    assert isinstance(called_messages, list)
    assert isinstance(called_messages[0], HumanMessage)
    assert called_messages[0].content == "今天吃什么？"

    assert "messages" in result
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][0], HumanMessage)
    assert isinstance(result["messages"][1], AIMessage)
    assert result["messages"][1].content == "今天可以吃牛肉面"


@pytest.mark.asyncio
async def test_graph_executes_tool_call_and_logs_tool_lifecycle(caplog) -> None:
    """模型返回 tool_calls 时，graph 应该执行工具并记录工具生命周期日志。"""

    model_runtime = ToolCallingAgentModelRuntime()
    graph = build_graph(model_runtime=model_runtime)

    with caplog.at_level(logging.INFO, logger="app.agent.nodes.tools"):
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="检查一下工具系统")],
                "model_options": None,
                "thread_id": "thread-tool-test",
                "user_id": "user-1",
            }
        )

    assert len(model_runtime.chat_model.calls) == 2
    assert any(isinstance(message, ToolMessage) for message in result["messages"])
    assert result["messages"][-1].content == "工具系统正常"
    assert "agent.tool.started tool_name=health_check" in caplog.text
    assert "agent.tool.completed tool_name=health_check" in caplog.text
    assert "thread_id=thread-tool-test" in caplog.text
