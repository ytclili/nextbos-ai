from langchain_core.messages import SystemMessage

from app.agent.prompts.renderer import build_system_message, load_system_prompt


def test_load_system_prompt_reads_shoudanba_business_prompt() -> None:
    """系统提示词应该从 Markdown 文件读取收单吧业务设定。"""

    prompt = load_system_prompt()

    assert "收单吧" in prompt
    assert "订单 + 回款 + 对账 + 账期风控 + AI 经营报表" in prompt
    assert "隐私与安全边界" in prompt


def test_build_system_message_returns_langchain_system_message() -> None:
    """renderer 应该把系统提示词转成 LangChain SystemMessage。"""

    message = build_system_message()

    assert isinstance(message, SystemMessage)
    assert "收单吧" in message.content
