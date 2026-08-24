import logging

from fastapi import FastAPI
from opentelemetry.sdk.resources import Resource

from app.core import tracing
from app.core.config import Settings


class FakeTracerProvider:
    """测试用 trace provider，只记录被挂载的 span processor。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.span_processors = []

    def add_span_processor(self, processor):
        self.span_processors.append(processor)


class FakeInstrumentor:
    """测试用 instrumentor，避免单元测试真的改全局 OTel instrumentation。"""

    def instrument(self, **_kwargs):
        return None


class FakeLogExporter:
    """记录 OTLP 日志 exporter 的初始化参数。"""

    instances = []

    def __init__(self, *, endpoint, headers):
        self.endpoint = endpoint
        self.headers = headers
        self.instances.append(self)


class FakeLogRecordProcessor:
    """记录日志 processor 使用的 exporter。"""

    def __init__(self, exporter, **kwargs):
        self.exporter = exporter
        self.kwargs = kwargs


class FakeLoggerProvider:
    """测试用 logger provider，只记录被挂载的 log processor。"""

    instances = []

    def __init__(self, *, resource):
        self.resource = resource
        self.log_processors = []
        self.instances.append(self)

    def add_log_record_processor(self, processor):
        self.log_processors.append(processor)

    def shutdown(self):
        return None


class FakeLoggingHandler(logging.Handler):
    """测试用 OTel logging handler，避免真实发送日志。"""

    instances = []

    def __init__(self, *, level, logger_provider):
        super().__init__(level=level)
        self.logger_provider = logger_provider
        self.instances.append(self)

    def emit(self, record):
        return None


def test_configure_tracing_enables_otlp_logging_only_when_logs_switch_is_on(monkeypatch):
    """OTEL_LOGS_ENABLED=true 时才把日志 handler 挂到 root logger。"""

    _patch_otel_tracing_side_effects(monkeypatch)
    calls = []
    monkeypatch.setattr(tracing, "_configure_otlp_logging", lambda **kwargs: calls.append(kwargs))

    tracing.configure_tracing(
        FastAPI(),
        Settings(
            otel_enabled=True,
            otel_logs_enabled=False,
            otel_exporter_otlp_endpoint="http://localhost:4317",
        ),
    )
    assert calls == []

    tracing.configure_tracing(
        FastAPI(),
        Settings(
            otel_enabled=True,
            otel_logs_enabled=True,
            otel_exporter_otlp_endpoint="http://localhost:4317",
        ),
    )
    assert len(calls) == 1


def test_configure_otlp_logging_adds_handler_with_configured_endpoint(monkeypatch):
    """日志导出器应该使用配置里的 OTLP endpoint 和 headers。"""

    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    FakeLogExporter.instances = []
    FakeLoggerProvider.instances = []
    FakeLoggingHandler.instances = []

    monkeypatch.setattr(tracing, "OTLPLogExporter", FakeLogExporter)
    monkeypatch.setattr(tracing, "BatchLogRecordProcessor", FakeLogRecordProcessor)
    monkeypatch.setattr(tracing, "LoggerProvider", FakeLoggerProvider)
    monkeypatch.setattr(tracing, "LoggingHandler", FakeLoggingHandler)
    monkeypatch.setattr(tracing, "set_logger_provider", lambda _provider: None)
    monkeypatch.setattr(tracing, "_logger_provider", None)

    try:
        tracing._configure_otlp_logging(
            settings=Settings(
                log_level="WARNING",
                otel_exporter_otlp_endpoint="http://localhost:4317",
            ),
            resource=Resource.create({"service.name": "nextbos-ai"}),
            headers={"tenant": "local"},
        )

        exporter = FakeLogExporter.instances[0]
        provider = FakeLoggerProvider.instances[0]
        handler = FakeLoggingHandler.instances[0]

        assert exporter.endpoint == "http://localhost:4317"
        assert exporter.headers == {"tenant": "local"}
        assert provider.log_processors[0].exporter is exporter
        assert provider.log_processors[0].kwargs == {"schedule_delay_millis": 1000}
        assert handler.level == logging.WARNING
        assert handler in root_logger.handlers
    finally:
        root_logger.handlers = original_handlers
        tracing._logger_provider = None


def test_configure_otlp_logging_does_not_add_duplicate_handler(monkeypatch):
    """重复初始化时不应该重复挂 OTel logs handler。"""

    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    FakeLogExporter.instances = []
    FakeLoggerProvider.instances = []
    FakeLoggingHandler.instances = []

    monkeypatch.setattr(tracing, "OTLPLogExporter", FakeLogExporter)
    monkeypatch.setattr(tracing, "BatchLogRecordProcessor", FakeLogRecordProcessor)
    monkeypatch.setattr(tracing, "LoggerProvider", FakeLoggerProvider)
    monkeypatch.setattr(tracing, "LoggingHandler", FakeLoggingHandler)
    monkeypatch.setattr(tracing, "set_logger_provider", lambda _provider: None)
    monkeypatch.setattr(tracing, "_logger_provider", None)

    try:
        settings = Settings(
            log_level="INFO",
            otel_exporter_otlp_endpoint="http://localhost:4317",
        )
        resource = Resource.create({"service.name": "nextbos-ai"})

        tracing._configure_otlp_logging(settings=settings, resource=resource, headers=None)
        tracing._configure_otlp_logging(settings=settings, resource=resource, headers=None)

        otlp_handlers = [
            handler
            for handler in root_logger.handlers
            if getattr(handler, tracing._OTLP_LOG_HANDLER_MARKER, False)
        ]
        assert len(otlp_handlers) == 1
        assert len(FakeLoggingHandler.instances) == 1
    finally:
        root_logger.handlers = original_handlers
        tracing._logger_provider = None


def _patch_otel_tracing_side_effects(monkeypatch):
    """屏蔽 configure_tracing 里会影响全局进程的 OTel 操作。"""

    monkeypatch.setattr(tracing, "TracerProvider", FakeTracerProvider)
    monkeypatch.setattr(tracing, "OTLPSpanExporter", lambda **_kwargs: object())
    monkeypatch.setattr(tracing, "BatchSpanProcessor", lambda exporter: exporter)
    monkeypatch.setattr(tracing.trace, "set_tracer_provider", lambda _provider: None)
    monkeypatch.setattr(
        tracing.FastAPIInstrumentor,
        "instrument_app",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(tracing, "SQLAlchemyInstrumentor", lambda: FakeInstrumentor())
    monkeypatch.setattr(tracing, "LoggingInstrumentor", lambda: FakeInstrumentor())
