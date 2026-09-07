from langchain_core.tools import BaseTool

from app.tools.builtin.health import health_check
from app.tools.builtin.memory import manage_memory, search_memory


def get_builtin_tools() -> list[BaseTool]:
    """返回 agent 默认可用的内置工具。

    这里统一收口工具列表，避免 LangGraph 节点里到处 import 具体工具。

    health_check 是项目自己的健康检查工具。
    search_memory / manage_memory 是 LangMem 官方长期记忆工具。
    """

    return [
        health_check,
        search_memory,
        manage_memory,
    ]


def get_builtin_tool_names() -> list[str]:
    """返回内置工具名称列表，主要用于日志、调试和测试。"""

    return sorted(tool.name for tool in get_builtin_tools())