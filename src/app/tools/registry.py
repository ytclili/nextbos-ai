from langchain_core.tools import BaseTool

from app.tools.builtin.health import health_check


def get_builtin_tools() -> list[BaseTool]:
    """返回 agent 默认可用的内置工具。

    这里统一收口工具列表，避免 LangGraph 节点里到处 import 具体工具。
    后续新增工具时，只需要在这里加入列表。

    第一版先只放无副作用的健康检查工具。
    """

    return [
        health_check,
    ]


def get_builtin_tool_names() -> list[str]:
    """返回内置工具名称列表，主要用于日志、调试和测试。"""

    return sorted(tool.name for tool in get_builtin_tools())