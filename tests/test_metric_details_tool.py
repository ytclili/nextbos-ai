from app.core.config import Settings
from app.tools.business import metric_details as metric_details_module
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
    response = FakeResponse({"data": {"items": [{"orderNo": "SO-001"}], "total": 1}})

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


async def test_get_metric_details_calls_internal_api_with_query_and_token(monkeypatch) -> None:
    """get_metric_details 应该带指标、分页和 mock token 查询经营指标明细。"""

    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeResponse({"data": {"items": [{"orderNo": "SO-001"}]}})

    monkeypatch.setattr(metric_details_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        metric_details_module,
        "get_settings",
        lambda: Settings(
            core_internal_base_url="http://core.internal",
            core_internal_token="mock-token",
            core_internal_timeout_seconds=4.5,
        ),
    )

    result = await metric_details_module.get_metric_details.ainvoke(
        {
            "metric": "pending",
            "period": "yesterday",
            "page": 2,
            "page_size": 25,
            "sort_by": "amount_desc",
        }
    )

    assert result == {"items": [{"orderNo": "SO-001"}]}
    assert FakeAsyncClient.response.raise_for_status_called is True
    assert FakeAsyncClient.requests == [
        {
            "url": (
                "http://core.internal/api/ai-tools/metric-details?"
                "metric=pending&period=yesterday&page=2&pageSize=25&sortBy=amount_desc"
            ),
            "headers": {"Authorization": "Bearer mock-token"},
            "timeout": 4.5,
        }
    ]


async def test_get_metric_details_omits_sort_by_when_missing(monkeypatch) -> None:
    """sort_by 为空时，请求不应该拼接 sortBy 参数。"""

    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeResponse({"data": {"items": []}})

    monkeypatch.setattr(metric_details_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        metric_details_module,
        "get_settings",
        lambda: Settings(
            core_internal_base_url="http://core.internal",
            core_internal_token="Bearer existing-token",
        ),
    )

    result = await metric_details_module.get_metric_details.ainvoke({"metric": "received"})

    assert result == {"items": []}
    assert FakeAsyncClient.requests == [
        {
            "url": (
                "http://core.internal/api/ai-tools/metric-details?"
                "metric=received&period=yesterday&page=1&pageSize=50"
            ),
            "headers": {"Authorization": "Bearer existing-token"},
            "timeout": 10.0,
        }
    ]


async def test_get_metric_details_returns_empty_dict_when_response_has_no_content(
    monkeypatch,
) -> None:
    """内部接口没有响应体时，get_metric_details 应该返回空字典。"""

    FakeAsyncClient.requests = []
    FakeAsyncClient.response = FakeResponse(None)

    monkeypatch.setattr(metric_details_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(
        metric_details_module,
        "get_settings",
        lambda: Settings(
            core_internal_base_url="http://core.internal",
            core_internal_token="mock-token",
        ),
    )

    result = await metric_details_module.get_metric_details.ainvoke({"metric": "receivable"})

    assert result == {}


def test_builtin_tool_registry_contains_metric_details_tool() -> None:
    """默认工具注册表应该包含经营指标明细工具。"""

    assert "get_metric_details" in get_builtin_tool_names()
