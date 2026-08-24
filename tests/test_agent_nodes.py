import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes.respond import create_respond_node
from app.llm.ports import ChatCompletion


class FakeAgentModelRuntime:
    """测试用模型运行时。

    这里不连接数据库，也不调用真实大模型。
    它只记录 respond 节点传进来的参数，然后返回一个固定的假回复。
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def chat(
        self,
        *,
        messages: list[object],
        options: object | None,
    ) -> ChatCompletion:
        """模拟 AgentModelRuntime.chat()。"""

        self.calls.append(
            {
                "messages": messages,
                "options": options,
            }
        )

        return ChatCompletion(content="假模型回复")


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

    assert len(model_runtime.calls) == 1

    saved_messages = model_runtime.calls[0]["messages"]
    assert isinstance(saved_messages, list)
    assert isinstance(saved_messages[0], HumanMessage)
    assert saved_messages[0].content == "今天吃什么？"

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "假模型回复"
