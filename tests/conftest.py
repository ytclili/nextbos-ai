import os

import pytest

from app.core.config import get_settings

os.environ["OTEL_ENABLED"] = "false"


@pytest.fixture(autouse=True)
def disable_tracing_for_tests(monkeypatch):
    """单元测试默认关闭 OpenTelemetry exporter，避免依赖本地 SigNoZ。"""

    monkeypatch.setenv("OTEL_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
