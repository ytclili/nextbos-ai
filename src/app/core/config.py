from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "nextbos-ai"
    app_env: str = "dev"
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    redis_url: str = "redis://localhost:6379/0"
    redis_checkpoint_ttl_seconds: int = 604800
    postgres_dsn: str = "postgresql+asyncpg://agent:agent@localhost:5432/agent_runtime"

    # LLM 运行时配置：.env 中的是默认值/兜底值，后续可被后台动态配置覆盖
    llm_provider: str = "none"  # none=不启用 LLM（回显模式）；openai=OpenAI 兼容接口
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_temperature: float = 0.7
    llm_timeout_seconds: int = 60

    # LangMem 短期上下文压缩配置。
    # 超过 trigger 后触发 rolling summary，最终模型输入尽量控制在 max_tokens 内。
    summary_max_tokens: int = 8000
    summary_trigger_tokens: int = 6000
    summary_max_output_tokens: int = 800

    # 后台动态配置落库时对 api_key 做应用层加密的密钥（预留）
    config_encryption_key: str = ""

    # OpenTelemetry / SigNoZ 调用链路配置。
    # 第一版默认关闭，避免本地没有 SigNoZ/Collector 时影响服务启动。
    otel_enabled: bool = False

    # 本地调试用：把 span 直接打印到终端。
    # 没有启动 SigNoZ 时，可以先打开这个，马上看到 trace 输出。
    otel_console_exporter_enabled: bool = False

    # 是否把 Python logging 日志通过 OTLP 发送到 SigNoZ Logs。
    # Trace 跑通后再单独打开它，避免本地开发时无意间产生大量日志写入。
    otel_logs_enabled: bool = False

    # trace / log 里的服务名。SigNoZ 里会按这个名字展示服务。
    otel_service_name: str = "nextbos-ai"

    # OTLP endpoint。自建 SigNoZ 常见值：
    # - gRPC: http://localhost:4317
    # - HTTP: http://localhost:4318
    # 当前项目使用 gRPC exporter。
    otel_exporter_otlp_endpoint: str = ""

    # OTLP headers。SigNoZ Cloud 可能需要 token；本地 SigNoZ 一般不用。
    # 注意：这个字段可能包含密钥，不要打印到日志。
    otel_exporter_otlp_headers: str = ""

    # trace 采样率。1.0 = 全采样；生产高流量时可以调低。
    otel_traces_sample_rate: float = Field(default=1.0, ge=0, le=1)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
