import logging

NOISY_LIBRARY_LOGGERS = {
    # Redis checkpoint 启动时会重复打印索引已存在、standalone client 等信息。
    "redisvl.index.index": logging.WARNING,
    "langgraph.checkpoint.redis.aio": logging.WARNING,
    # LangChain/OpenAI/httpx 的 INFO 更多是 SDK 诊断，对业务排障意义不大。
    "langchain_openai": logging.WARNING,
    "httpx": logging.WARNING,
}


class OpenTelemetryLogFilter(logging.Filter):
    """给未启用 OTel 的日志补默认字段。

    LoggingInstrumentor 启用后，会给日志记录自动注入 otelTraceID / otelSpanID。
    但如果 OTel 没启用，或者日志发生在 span 外，这些字段可能不存在。
    这里补默认值，避免 logging format 因字段缺失报错。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "otelTraceID"):
            record.otelTraceID = "-"
        if not hasattr(record, "otelSpanID"):
            record.otelSpanID = "-"
        return True


def _install_otel_safe_log_record_factory() -> None:
    """给所有日志记录安装 OTel 字段兜底。

    root logger 的 filter 不一定会处理第三方库创建的 LogRecord。
    这里从 LogRecordFactory 层统一补字段，保证 uvicorn、Redis、SQLAlchemy
    等库的日志也能安全使用同一套 format。
    """

    current_factory = logging.getLogRecordFactory()
    if getattr(current_factory, "_nextbos_otel_safe", False):
        return

    def safe_factory(*args, **kwargs):
        record = current_factory(*args, **kwargs)
        if not hasattr(record, "otelTraceID"):
            record.otelTraceID = "-"
        if not hasattr(record, "otelSpanID"):
            record.otelSpanID = "-"
        return record

    safe_factory._nextbos_otel_safe = True
    logging.setLogRecordFactory(safe_factory)


def configure_logging(level: str = "INFO") -> None:
    _install_otel_safe_log_record_factory()
    _configure_noisy_library_loggers()

    root_logger = logging.getLogger()
    root_logger.addFilter(OpenTelemetryLogFilter())

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=(
            "%(asctime)s %(levelname)s %(name)s "
            "trace_id=%(otelTraceID)s span_id=%(otelSpanID)s "
            "%(message)s"
        ),
    )

    for handler in root_logger.handlers:
        handler.addFilter(OpenTelemetryLogFilter())


def _configure_noisy_library_loggers() -> None:
    """把高噪声第三方库日志降级到 WARNING。

    业务日志继续保留 INFO；Redis、LangChain、httpx 这类库的正常运行诊断
    默认不进入 SigNoZ Logs，避免生产日志列表被低价值信息刷屏。
    """

    for logger_name, logger_level in NOISY_LIBRARY_LOGGERS.items():
        logging.getLogger(logger_name).setLevel(logger_level)
