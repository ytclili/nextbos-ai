import logging

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from app.agent.graph import build_graph
from app.agent.schemas.intent import IntentDecision
from app.llm.models import EffectiveModelConfig, ProviderCredential


class FakeAgentModelRuntime:
    """测试用模型运行时。

    这个假对象用来替代真实 AgentModelRuntime。
    这样测试 graph 的时候不会访问数据库，也不会请求真实大模型。
    """

    def __init__(self) -> None:
        self.resolve_calls: list[object | None] = []
        self.intent_decision = _direct_answer_intent()
        self.chat_model = FakeChatModel(self.intent_decision)

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

    def __init__(self, intent_decision: IntentDecision | None = None) -> None:
        self.calls: list[list[object]] = []
        self.structured_calls: list[list[object]] = []
        self.intent_decision = intent_decision or _direct_answer_intent()

    def with_structured_output(self, _schema):
        return FakeStructuredChatModel(self)

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content="今天可以吃牛肉面")


class FakeStructuredChatModel:
    """测试用结构化输出模型，只服务 intent 节点。"""

    def __init__(self, chat_model: FakeChatModel) -> None:
        self.chat_model = chat_model

    async def ainvoke(self, messages):
        self.chat_model.structured_calls.append(messages)
        return self.chat_model.intent_decision


class ToolCallingAgentModelRuntime(FakeAgentModelRuntime):
    """先请求调用工具，再基于工具结果返回最终回复。"""

    def __init__(self) -> None:
        super().__init__()
        self.chat_model = ToolCallingChatModel(self.intent_decision)


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


class MemoryManagingAgentModelRuntime(FakeAgentModelRuntime):
    """先请求写入长期记忆，再返回最终回复。"""

    def __init__(self) -> None:
        super().__init__()
        self.chat_model = MemoryManagingChatModel(self.intent_decision)


class MemoryManagingChatModel(FakeChatModel):
    """模拟模型调用 LangMem manage_memory 工具。"""

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "manage_memory",
                        "args": {
                            "content": "用户喜欢粤菜",
                            "action": "create",
                        },
                        "id": "call-manage-memory",
                    }
                ],
            )
        return AIMessage(content="我已经记住了。")


class MemorySearchingAgentModelRuntime(FakeAgentModelRuntime):
    """先请求搜索长期记忆，再基于搜索结果返回最终回复。"""

    def __init__(self) -> None:
        super().__init__()
        self.chat_model = MemorySearchingChatModel(self.intent_decision)


class MemorySearchingChatModel(FakeChatModel):
    """模拟模型调用 LangMem search_memory 工具。"""

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_memory",
                        "args": {
                            "query": "用户饮食偏好",
                            "limit": 10,
                        },
                        "id": "call-search-memory",
                    }
                ],
            )
        return AIMessage(content="我查到你喜欢粤菜。")


class RepeatedToolAgentModelRuntime(FakeAgentModelRuntime):
    """模拟真实模型在工具可用时重复请求同一个工具。"""

    def __init__(self) -> None:
        super().__init__()
        self.chat_models: list[RepeatedToolChatModel] = []

    def create_chat_model(self, config: EffectiveModelConfig):
        chat_model = RepeatedToolChatModel(self.intent_decision)
        self.chat_models.append(chat_model)
        return chat_model


class RepeatedToolChatModel(FakeChatModel):
    """只要绑定了工具就继续 tool_call，没有工具时才输出最终文本。"""

    def __init__(self, intent_decision: IntentDecision | None = None) -> None:
        super().__init__(intent_decision)
        self.tools_bound = False

    def bind_tools(self, _tools):
        self.tools_bound = True
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if self.tools_bound:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "manage_memory",
                        "args": {
                            "content": "用户喜欢粤菜",
                            "action": "create",
                        },
                        "id": "call-repeated-manage-memory",
                    }
                ],
            )
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="我已经记住了。")
        return AIMessage(content="没有看到工具结果。")


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

    assert model_runtime.resolve_calls == [None, None]

    called_messages = model_runtime.chat_model.calls[0]
    assert isinstance(called_messages, list)
    assert isinstance(called_messages[0], SystemMessage)
    assert "AI 管家" in called_messages[0].content
    assert isinstance(called_messages[1], HumanMessage)
    assert called_messages[1].content == "今天吃什么？"

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


@pytest.mark.asyncio
async def test_graph_manage_memory_tool_writes_to_langgraph_store() -> None:
    """LangMem manage_memory 工具应该写入 graph 编译时传入的 Store。"""

    model_runtime = MemoryManagingAgentModelRuntime()
    store = InMemoryStore()
    graph = build_graph(model_runtime=model_runtime, store=store)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="请记住我喜欢粤菜")],
            "model_options": None,
            "thread_id": "thread-memory-tool-test",
            "user_id": "user-memory-test",
        },
        config={
            "configurable": {
                "thread_id": "thread-memory-tool-test",
                "langgraph_user_id": "user-memory-test",
            }
        },
    )

    memories = await store.asearch(("memories", "user-memory-test"), limit=10)

    assert result["messages"][-1].content == "我已经记住了。"
    assert any(memory.value["content"] == "用户喜欢粤菜" for memory in memories)


@pytest.mark.asyncio
async def test_graph_search_memory_tool_reads_from_langgraph_store() -> None:
    """LangMem search_memory 工具应该读取 graph 编译时传入的 Store。"""

    model_runtime = MemorySearchingAgentModelRuntime()
    store = InMemoryStore()
    await store.aput(
        ("memories", "user-search-test"),
        "memory-1",
        {"content": "用户喜欢粤菜"},
    )
    graph = build_graph(model_runtime=model_runtime, store=store)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="你记得我的饮食偏好吗？")],
            "model_options": None,
            "thread_id": "thread-search-memory-test",
            "user_id": "user-search-test",
        },
        config={
            "configurable": {
                "thread_id": "thread-search-memory-test",
                "langgraph_user_id": "user-search-test",
            }
        },
    )

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]

    assert result["messages"][-1].content == "我查到你喜欢粤菜。"
    assert any("用户喜欢粤菜" in str(message.content) for message in tool_messages)


@pytest.mark.asyncio
async def test_graph_uses_tool_free_final_response_after_tool_execution() -> None:
    """工具执行后应该用不带工具的模型生成最终回复，避免工具循环。"""

    model_runtime = RepeatedToolAgentModelRuntime()
    store = InMemoryStore()
    graph = build_graph(model_runtime=model_runtime, store=store)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="请记住我喜欢粤菜")],
            "model_options": None,
            "thread_id": "thread-repeated-tool-test",
            "user_id": "user-repeated-tool-test",
        },
        config={
            "configurable": {
                "thread_id": "thread-repeated-tool-test",
                "langgraph_user_id": "user-repeated-tool-test",
            },
            "recursion_limit": 8,
        },
    )

    assert result["messages"][-1].content == "我已经记住了。"
    assert len(model_runtime.chat_models) == 3
    assert model_runtime.chat_models[1].tools_bound is True
    assert model_runtime.chat_models[2].tools_bound is False


def _direct_answer_intent() -> IntentDecision:
    return IntentDecision(
        intent_type="direct_answer",
        needs_business_data=False,
        output_type="text",
        question_rewrite="直接回答用户问题",
        metrics=[],
        dimensions=[],
        filters=[],
        time_range=None,
        confidence=0.9,
        missing_slots=[],
        clarification_question=None,
    )
