from app.core.config import Settings
from app.tools.business import top_pending_customer as top_pending_customer_module
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
    response = FakeResponse({"data": {"customer": "客户 B", "pendingAmount": 500}})

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


async def test_get_top_pending_customer_calls_internal_api_with_period_and_token(
    monkeypatch,
) -> None:
    """get_top_pending_customer 应该带 period 和 token 调用待收最高客户接口。"""

    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeResponse(
        {"data": {"customer": "客户 B", "pendingAmount": 500}}
    )

    monkeypatch.setattr(top_pending_customer_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        top_pending_customer_module,
        "get_settings",
        lambda: Settings(
            core_internal_base_url="http://127.0.0.1:3000",
            core_internal_token="mock-token",
            core_internal_timeout_seconds=4.5,
        ),
    )

    result = await top_pending_customer_module.get_top_pending_customer.ainvoke({})

    assert result == {"customer": "客户 B", "pendingAmount": 500}
    assert FakeAsyncClient.response.raise_for_status_called is True
    assert FakeAsyncClient.requests == [
        {
            "url": (
                "http://127.0.0.1:3000/api/ai-tools/customers/top-pending?"
                "period=yesterday"
            ),
            "headers": {"Authorization": "Bearer mock-token"},
            "timeout": 4.5,
        }
    ]


async def test_get_top_pending_customer_returns_none_when_response_has_no_content(
    monkeypatch,
) -> None:
    """内部接口没有响应体时，get_top_pending_customer 应该返回 None。"""

    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeResponse(None)

    monkeypatch.setattr(top_pending_customer_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        top_pending_customer_module,
        "get_settings",
        lambda: Settings(
            core_internal_base_url="http://core.internal",
            core_internal_token="mock-token",
        ),
    )

    result = await top_pending_customer_module.get_top_pending_customer.ainvoke({})

    assert result is None


async def test_get_top_pending_customer_keeps_existing_bearer_token(monkeypatch) -> None:
    """CORE_INTERNAL_TOKEN 已带 Bearer 时，工具不应该重复拼接认证前缀。"""

    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeResponse({"data": None})

    monkeypatch.setattr(top_pending_customer_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        top_pending_customer_module,
        "get_settings",
        lambda: Settings(
            core_internal_base_url="http://core.internal",
            core_internal_token="Bearer existing-token",
        ),
    )

    await top_pending_customer_module.get_top_pending_customer.ainvoke(
        {"period": "yesterday"}
    )

    assert FakeAsyncClient.requests[0]["headers"] == {"Authorization": "Bearer existing-token"}


def test_builtin_tool_registry_contains_top_pending_customer_tool() -> None:
    """默认工具注册表应该包含待收最高客户工具。"""

    assert "get_top_pending_customer" in get_builtin_tool_names()
