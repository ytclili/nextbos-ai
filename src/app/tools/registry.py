from langchain_core.tools import BaseTool

from app.tools.builtin.health import health_check
from app.tools.builtin.memory import manage_memory, search_memory
from app.tools.business.dashboard import get_dashboard
from app.tools.business.metric_details import get_metric_details
from app.tools.business.top_pending_customer import get_top_pending_customer


def get_builtin_tools() -> list[BaseTool]:
    """返回 agent 默认可用的内置工具。

    这里统一收口工具列表，避免 LangGraph 节点里到处 import 具体工具。

    health_check 是项目自己的健康检查工具。
    search_memory / manage_memory 是 LangMem 官方长期记忆工具。
    get_dashboard 是收单吧业务经营看板查询工具。
    get_metric_details 是收单吧经营指标明细查询工具。
    get_top_pending_customer 是收单吧待收款最高客户查询工具。
    """

    return [
        health_check,
        get_dashboard,
        get_metric_details,
        get_top_pending_customer,
        search_memory,
        manage_memory,
    ]


def get_builtin_tool_names() -> list[str]:
    """返回内置工具名称列表，主要用于日志、调试和测试。"""

    return sorted(tool.name for tool in get_builtin_tools())
