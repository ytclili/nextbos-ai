from langchain_core.tools import tool


@tool
async def health_check() -> str:
    """检查 agent 工具系统是否可用。

    当用户询问工具系统、后端状态、健康检查是否正常时，可以调用这个工具。
    这个工具没有业务副作用，只返回一个简单的健康状态。
    """

    return "agent tool system is healthy"