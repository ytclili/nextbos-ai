import json
from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import DEPLOYMENT_ENVIRONMENT, SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

from app.core.config import Settings

OtelScalarAttribute = bool | str | bytes | int | float
OtelAttributeValue = OtelScalarAttribute | Sequence[OtelScalarAttribute]


def configure_tracing(app: FastAPI, settings: Settings) -> None:
    """初始化 OpenTelemetry 调用链路。

    这个函数负责把 FastAPI 请求、SQLAlchemy 数据库访问等基础 span 接起来。
    开启 console exporter 后，本地不用 SigNoZ 也能直接在终端看到 span。
    配置 OTLP endpoint 后，trace 会发送到 SigNoZ / OpenTelemetry Collector。
    """

    if not settings.otel_enabled:
        return

    tracer_provider = TracerProvider(
        resource=Resource.create(
            {
                SERVICE_NAME: settings.otel_service_name,
                DEPLOYMENT_ENVIRONMENT: settings.app_env,
            }
        ),
        sampler=TraceIdRatioBased(settings.otel_traces_sample_rate),
    )

    if settings.otel_console_exporter_enabled:
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    if settings.otel_exporter_otlp_endpoint:
        span_exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            headers=_parse_otlp_headers(settings.otel_exporter_otlp_headers),
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

    trace.set_tracer_provider(tracer_provider)

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
