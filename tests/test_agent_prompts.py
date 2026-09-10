from langchain_core.messages import SystemMessage

from app.agent.prompts.renderer import build_system_message, load_system_prompt

SQL_PLAN_PROMPT_PATH = "src/app/agent/prompts/sql_plan.md"


def test_load_system_prompt_reads_shop_wms_business_prompt() -> None:
    """系统提示词应该从 Markdown 文件读取 Shop 和 WMS 业务设定。"""

    prompt = load_system_prompt()

    assert "AI 管家" in prompt
    assert "Shop 小程序" in prompt
    assert "WMS 仓储" in prompt
    assert "隐私与安全边界" in prompt


def test_build_system_message_returns_langchain_system_message() -> None:
    """renderer 应该把系统提示词转成 LangChain SystemMessage。"""

    message = build_system_message()

    assert isinstance(message, SystemMessage)
    assert "AI 管家" in message.content


def test_sql_plan_prompt_defines_safe_planning_boundary() -> None:
    """SQL planning 提示词应该明确只规划、不执行、必须经过 dry-run。"""

    prompt = open(SQL_PLAN_PROMPT_PATH, encoding="utf-8").read()

    assert "SQL 规划节点" in prompt
    assert "不执行 SQL" in prompt
    assert "不能跳过 Wren dry-plan / dry-run" in prompt
    assert "insert、update、delete、drop、alter、truncate" in prompt
