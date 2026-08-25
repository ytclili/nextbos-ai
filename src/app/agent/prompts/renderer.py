from functools import lru_cache
from pathlib import Path

from langchain_core.messages import SystemMessage

PROMPTS_DIR = Path(__file__).parent
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system.md"


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    """读取 agent 系统提示词。

    系统提示词属于代码级行为契约，第一版直接用 Git 管理的 Markdown 文件。
    这里加一层很薄的读取函数，避免 respond 节点里散落文件路径。
    """

    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_system_message() -> SystemMessage:
    """构造 LangChain SystemMessage。"""

    return SystemMessage(content=load_system_prompt())


def prepend_system_prompt(messages: list[object]) -> list[object]:
    """把系统提示词放到模型输入首位。

    注意：这里返回新 list，不修改传入的 messages。
    这样 system prompt 只参与本次模型输入，不会被写入 LangGraph state / Redis checkpoint。
    """

    return [build_system_message(), *messages]
