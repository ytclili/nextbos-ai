from functools import lru_cache

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

    # 后台动态配置落库时对 api_key 做应用层加密的密钥（预留）
    config_encryption_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
