from fastapi import HTTPException

from app.core.errors import InfrastructureUnavailableError
from app.llm.errors import LLMConfigurationError, LLMError
from app.tools.errors import ToolError, ToolTimeoutError


def map_exception_to_http_error(exc: Exception) -> HTTPException:
    """把应用内部异常映射成前端可识别的 HTTP 错误。

    这里统一输出结构化 detail：
    {
        "code": "...",
        "message": "..."
    }

    路由层只负责调用这个函数，不在每个接口里散落异常判断。
    """

    if isinstance(exc, LLMConfigurationError):
        return _http_error(
            status_code=500,
            code="llm_configuration_error",
            message="模型配置不可用，请检查模型、请求地址或 API Key。",
        )

    if isinstance(exc, ToolTimeoutError):
        return _http_error(
            status_code=504,
            code="tool_timeout",
            message="工具执行超时，请稍后重试。",
        )

    if _is_timeout_error(exc):
        return _http_error(
            status_code=504,
            code="llm_timeout",
            message="模型请求超时，请稍后重试或缩短输入。",
        )

    if isinstance(exc, ToolError):
        return _http_error(
            status_code=502,
            code="tool_error",
            message="工具执行失败，请稍后重试。",
        )

    if isinstance(exc, LLMError):
        return _http_error(
            status_code=502,
            code="llm_error",
            message="模型服务调用失败，请稍后重试。",
        )

    if isinstance(exc, InfrastructureUnavailableError):
        return _http_error(
            status_code=503,
            code="infrastructure_unavailable",
            message="Agent 依赖的基础设施暂时不可用，请稍后重试。",
        )

    return _http_error(
        status_code=500,
        code="internal_error",
        message="服务内部错误，请稍后重试。",
    )


def _http_error(*, status_code: int, code: str, message: str) -> HTTPException:
    """创建统一结构的 HTTPException。"""

    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
        },
    )


def _is_timeout_error(exc: Exception) -> bool:
    """识别常见 timeout 异常。

    OpenAI / httpx / LangChain 的 timeout 异常类型可能不同；
    第一版先覆盖 Python TimeoutError 和常见错误文案，避免把 timeout 错归为 503。
    """

    if isinstance(exc, TimeoutError):
        return True

    message = str(exc).lower()
    return "timed out" in message or "timeout" in message
