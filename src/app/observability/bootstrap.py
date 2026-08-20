"""OpenTelemetry 启动入口（预留）。

后续在这里集中初始化 TracerProvider、MeterProvider、OTLP exporter，
并注册 FastAPI、HTTPX、Redis、SQLAlchemy 等官方自动埋点。
该模块不应包含业务 Span，也不应被业务模块重复初始化。
"""

# TODO: 实现 setup_observability(app, settings)；当前不执行任何初始化。
