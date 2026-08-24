import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import build_graph
from app.llm.ports import ChatCompletion


class FakeAgentModelRuntime:
    """测试用模型运行时。

    这个假对象用来替代真实 AgentModelRuntime。
    这样测试 graph 的时候不会访问数据库，也不会请求真实大模型。
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def chat(
        self,
        *,
        messages: list[object],
        options: object | None,
    ) -> ChatCompletion:
        """模拟模型调用，并记录 graph 传进来的参数。"""

        self.calls.append(
            {
                "messages": messages,
                "options": options,
            }
        )

        return ChatCompletion(content="今天可以吃牛肉面")


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

    assert len(model_runtime.calls) == 1

    called_messages = model_runtime.calls[0]["messages"]
    assert isinstance(called_messages, list)
    assert isinstance(called_messages[0], HumanMessage)
    assert called_messages[0].content == "今天吃什么？"

    assert "messages" in result
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][0], HumanMessage)
    assert isinstance(result["messages"][1], AIMessage)
    assert result["messages"][1].content == "今天可以吃牛肉面"
