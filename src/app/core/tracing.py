import json
import logging
from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import DEPLOYMENT_ENVIRONMENT, SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from app.core.config import Settings

OtelScalarAttribute = bool | str | bytes | int | float
OtelAttributeValue = OtelScalarAttribute | Sequence[OtelScalarAttribute]
_logger_provider: LoggerProvider | None = None
_OTLP_LOG_HANDLER_MARKER = "_nextbos_otlp_log_handler"
_OTLP_LOG_HANDLER_PROVIDER = "_nextbos_logger_provider"


def configure_tracing(app: FastAPI, settings: Settings) -> None:
    """初始化 OpenTelemetry 调用链路和日志导出。

    这个函数负责三件事：
    1. 创建 trace provider，把 FastAPI、SQLAlchemy、业务 span 接起来；
    2. 可选把 trace 发送到 SigNoZ / OpenTelemetry Collector；
    3. 可选把 Python logging 日志发送到 SigNoZ Logs。

    注意：终端日志由 app.core.logging 负责；
    这里的 logs exporter 负责“额外发一份到 SigNoZ”。
    """

    if not settings.otel_enabled:
        return

    resource = Resource.create(
        {
            SERVICE_NAME: settings.otel_service_name,
            DEPLOYMENT_ENVIRONMENT: settings.app_env,
        }
    )
    headers = _parse_otlp_headers(settings.otel_exporter_otlp_headers)

    tracer_provider = TracerProvider(
        resource=resource,
        sampler=TraceIdRatioBased(settings.otel_traces_sample_rate),
    )

    if settings.otel_console_exporter_enabled:
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    if settings.otel_exporter_otlp_endpoint:
        span_exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            headers=headers,
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

    trace.set_tracer_provider(tracer_provider)

    if settings.otel_logs_enabled and settings.otel_exporter_otlp_endpoint:
        _configure_otlp_logging(settings=settings, resource=resource, headers=headers)

    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
    SQLAlchemyInstrumentor().instrument(
        tracer_provider=tracer_provider,
    )
    LoggingInstrumentor().instrument(
        set_logging_format=False,
    )


def get_tracer(name: str):
    """获取业务代码里手动创建 span 用的 tracer。"""

    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """关闭 OTel 日志 provider，确保进程退出前把批量日志 flush 出去。"""

    global _logger_provider
    root_logger = logging.getLogger()
    removed_providers = _remove_existing_otlp_log_handlers(root_logger)

    if _logger_provider is not None and all(
        _logger_provider is not provider for provider in removed_providers
    ):
        _logger_provider.shutdown()
    _logger_provider = None


def set_span_attribute(span: Any, key: str, value: object) -> None:
    """安全写入 span attribute。

    OpenTelemetry attribute 只能接收字符串、数字、布尔、bytes，或者这些类型的数组。
    LangChain 的 usage_metadata 里可能包含 dict，例如 token_details。
    这里统一把复杂对象转成 JSON 字符串，避免 SDK 打出 Invalid type 警告。
    """

    if value is None:
        return

    span.set_attribute(key, _to_otel_attribute_value(value))


def set_span_attributes(span: Any, prefix: str, attributes: dict[str, object]) -> None:
    """批量写入带统一前缀的 span attribute。"""

    for key, value in attributes.items():
        set_span_attribute(span, f"{prefix}.{key}", value)


def _configure_otlp_logging(
    *,
    settings: Settings,
    resource: Resource,
    headers: dict[str, str] | None,
) -> None:
    """把 Python logging 日志导出到 SigNoZ Logs。

    这里会在 root logger 上额外挂一个 OTel LoggingHandler。
    原来的终端 StreamHandler 不会被替换，所以本地终端仍然能看到日志；
    只是当 OTEL_LOGS_ENABLED=true 时，日志会再发送一份到 OTLP endpoint。
    """

    global _logger_provider

    root_logger = logging.getLogger()
    if _logger_provider is not None and _has_existing_otlp_log_handler(root_logger):
        return

    removed_providers = _remove_existing_otlp_log_handlers(root_logger)
    if _logger_provider is not None and all(
        _logger_provider is not provider for provider in removed_providers
    ):
        _logger_provider.shutdown()
    _logger_provider = None

    logger_provider = LoggerProvider(resource=resource)
    _logger_provider = logger_provider
    set_logger_provider(logger_provider)

    logger_provider.add_log_record_processor(
        BatchLogRecordProcessor(
            OTLPLogExporter(
                endpoint=settings.otel_exporter_otlp_endpoint,
                headers=headers,
            ),
            # 本地开发时希望刷新 SigNoZ 后尽快看到日志；生产也仍然是批量发送。
            schedule_delay_millis=1000,
        )
    )

    otlp_handler = LoggingHandler(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        logger_provider=logger_provider,
    )
    setattr(otlp_handler, _OTLP_LOG_HANDLER_MARKER, True)
    setattr(otlp_handler, _OTLP_LOG_HANDLER_PROVIDER, logger_provider)
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    root_logger.addHandler(otlp_handler)


def _has_existing_otlp_log_handler(root_logger: logging.Logger) -> bool:
    """判断 root logger 上是否已有本项目的 OTel logs handler。"""

    return any(
        getattr(handler, _OTLP_LOG_HANDLER_MARKER, False)
        for handler in root_logger.handlers
    )


def _remove_existing_otlp_log_handlers(root_logger: logging.Logger) -> list[LoggerProvider]:
    """移除本项目之前挂载过的 OTel logs handler。

    uvicorn reload、测试进程复用、或应用重复初始化时，root logger 可能保留旧 handler。
    如果不清理，SigNoZ Logs 会看到同一条业务日志重复出现。
    """

    removed_providers = []
    for handler in list(root_logger.handlers):
        if not getattr(handler, _OTLP_LOG_HANDLER_MARKER, False):
            continue

        root_logger.removeHandler(handler)
        logger_provider = getattr(handler, _OTLP_LOG_HANDLER_PROVIDER, None)
        if logger_provider is not None:
            logger_provider.shutdown()
            removed_providers.append(logger_provider)
        handler.close()

    return removed_providers


def _to_otel_attribute_value(value: object) -> OtelAttributeValue:
    """把 Python 对象规整成 OTel SDK 可接受的 attribute 类型。"""

    if isinstance(value, bool | str | bytes | int | float):
        return value

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        normalized_items = [_to_otel_sequence_item(item) for item in value]
        if all(item is not None for item in normalized_items):
            return normalized_items

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _to_otel_sequence_item(value: object) -> OtelScalarAttribute | None:
    """规整 OTel 数组 attribute 的单个元素。"""

    if isinstance(value, bool | str | bytes | int | float):
        return value
    return None


def _parse_otlp_headers(raw_headers: str) -> dict[str, str] | None:
    """解析 OTLP header 配置。

    格式示例：
    OTEL_EXPORTER_OTLP_HEADERS=signoz-access-token=xxx,tenant=dev

    注意：header 里可能有 token，所以不要打印 raw_headers。
    """

    if not raw_headers:
        return None

    headers: dict[str, str] = {}
    for item in raw_headers.split(","):
        key, separator, value = item.partition("=")
        if separator and key.strip():
            headers[key.strip()] = value.strip()

    return headers or None
