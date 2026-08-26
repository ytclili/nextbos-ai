from app.core.config import Settings
from app.tools.business import dashboard as dashboard_module
from app.tools.business.auth import use_core_internal_token
from app.tools.registry import get_builtin_tool_names


class FakeResponse:
    """模拟 httpx 响应对象。"""

    def __init__(self, payload: dict | None = None, *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload) if payload is not None else ""
        self.content = b"{}" if payload is not None else b""
        self.raise_for_status_called = False

    def raise_for_status(self) -> None:
        """记录工具是否检查了 HTTP 状态码。"""

        self.raise_for_status_called = True

    def json(self) -> dict:
        """返回模拟接口 JSON。"""

        return self.payload or {}


class FakeAsyncClient:
    """模拟 httpx.AsyncClient，避免测试里真的发 HTTP 请求。"""

    requests: list[dict] = []
    response = FakeResponse({"data": {"received": 100, "pending": 20}})

    def __init__(self, *, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def get(self, url: str, *, headers: dict[str, str]):
        FakeAsyncClient.requests.append(
            {
                "url": url,
                "headers": headers,
                "timeout": self.timeout,
            }
        )
        return FakeAsyncClient.response


async def test_get_dashboard_calls_internal_api_with_period_and_token(monkeypatch) -> None:
    """get_dashboard 应该带 period 和 mock token 调用内部经营看板接口。"""

    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeResponse({"data": {"received": 100, "pending": 20}})

    monkeypatch.setattr(dashboard_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        dashboard_module,
        "get_settings",
        lambda: Settings(
            core_internal_base_url="http://core.internal",
            core_internal_token="mock-token",
            core_internal_timeout_seconds=3.5,
        ),
    )

    result = await dashboard_module.get_dashboard.ainvoke({"period": "today"})

    assert result == {"received": 100, "pending": 20}
    assert FakeAsyncClient.response.raise_for_status_called is True
    assert FakeAsyncClient.requests == [
        {
            "url": "http://core.internal/api/ai-tools/dashboard?period=today",
            "headers": {"Authorization": "Bearer mock-token"},
            "timeout": 3.5,
        }
    ]


async def test_get_dashboard_returns_empty_dict_when_response_has_no_content(
    monkeypatch,
) -> None:
    """内部接口没有响应体时，get_dashboard 应该返回空字典。"""

    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeResponse(None)

    monkeypatch.setattr(dashboard_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        dashboard_module,
        "get_settings",
        lambda: Settings(
            core_internal_base_url="http://core.internal",
            core_internal_token="mock-token",
        ),
    )

    result = await dashboard_module.get_dashboard.ainvoke({})

    assert result == {}


async def test_get_dashboard_keeps_existing_bearer_token(monkeypatch) -> None:
    """CORE_INTERNAL_TOKEN 已带 Bearer 时，工具不应该重复拼接认证前缀。"""

    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeResponse({"data": {"received": 100, "pending": 20}})

    monkeypatch.setattr(dashboard_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        dashboard_module,
        "get_settings",
        lambda: Settings(
            core_internal_base_url="http://core.internal",
            core_internal_token="Bearer existing-token",
        ),
    )

    await dashboard_module.get_dashboard.ainvoke({"period": "yesterday"})

    assert FakeAsyncClient.requests[0]["headers"] == {"Authorization": "Bearer existing-token"}


async def test_get_dashboard_prefers_runtime_token_override(monkeypatch) -> None:
    """有本次运行 token 时，业务工具应该优先使用它而不是 env token。"""

    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeResponse({"data": {"received": 100, "pending": 20}})

    monkeypatch.setattr(dashboard_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        dashboard_module,
        "get_settings",
        lambda: Settings(
            core_internal_base_url="http://core.internal",
            core_internal_token="env-token",
        ),
    )

    token = use_core_internal_token("resume-token")
    try:
        await dashboard_module.get_dashboard.ainvoke({"period": "yesterday"})
    finally:
        token.reset()

    assert FakeAsyncClient.requests[0]["headers"] == {"Authorization": "Bearer resume-token"}


def test_builtin_tool_registry_contains_dashboard_tool() -> None:
    """默认工具注册表应该包含经营看板工具。"""

    assert "get_dashboard" in get_builtin_tool_names()
