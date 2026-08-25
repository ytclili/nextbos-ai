from fastapi import HTTPException

from app.api.errors import map_exception_to_http_error
from app.core.errors import InfrastructureUnavailableError
from app.llm.errors import LLMConfigurationError, LLMError
from app.tools.errors import ToolTimeoutError


def test_map_llm_timeout_to_504() -> None:
    """LLM 超时应该返回 504，方便前端给出重试或缩短输入提示。"""

    error = map_exception_to_http_error(TimeoutError("Request timed out."))

    assert isinstance(error, HTTPException)
    assert error.status_code == 504
    assert error.detail == {
        "code": "llm_timeout",
        "message": "模型请求超时，请稍后重试或缩短输入。",
    }


def test_map_llm_configuration_error_to_500() -> None:
    """模型配置错误应该和基础设施错误区分开。"""

    error = map_exception_to_http_error(LLMConfigurationError("LLM api key is required"))

    assert error.status_code == 500
    assert error.detail == {
        "code": "llm_configuration_error",
        "message": "模型配置不可用，请检查模型、请求地址或 API Key。",
    }


def test_map_infrastructure_error_to_503() -> None:
    """Redis/PostgreSQL 等基础设施错误应该返回 503。"""

    error = map_exception_to_http_error(InfrastructureUnavailableError("redis unavailable"))

    assert error.status_code == 503
    assert error.detail == {
        "code": "infrastructure_unavailable",
        "message": "Agent 依赖的基础设施暂时不可用，请稍后重试。",
    }


def test_map_tool_timeout_to_504() -> None:
    """工具执行超时也应该返回 504。"""

    error = map_exception_to_http_error(ToolTimeoutError("tool timed out"))

    assert error.status_code == 504
    assert error.detail == {
        "code": "tool_timeout",
        "message": "工具执行超时，请稍后重试。",
    }


def test_map_unknown_llm_error_to_502() -> None:
    """未知 LLM 调用错误归类为上游模型错误。"""

    error = map_exception_to_http_error(LLMError("bad upstream response"))

    assert error.status_code == 502
    assert error.detail == {
        "code": "llm_error",
        "message": "模型服务调用失败，请稍后重试。",
    }
