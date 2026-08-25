import hashlib
import json
from dataclasses import dataclass, replace
from typing import Protocol

from app.core.config import Settings
from app.core.tracing import get_tracer
from app.llm.models import (
    EffectiveModelConfig,
    EffectiveModelConfigSnapshot,
    ModelProfile,
    ProviderCredential,
)

tracer = get_tracer(__name__)


@dataclass(frozen=True)
class ModelSelection:
    """调用方选择的模型身份。

    请求侧只允许选择数据库里登记过的 alias，不允许把任意 base_url 或 API key
    直接塞进运行链路。
    """

    model_alias: str | None = None


@dataclass(frozen=True)
class RequestedModelParams:
    """单次请求允许覆盖的生成参数。"""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    timeout_seconds: int | None = None

    def to_overrides(self) -> dict[str, object]:
        return {
            key: value
            for key, value in {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "timeout_seconds": self.timeout_seconds,
            }.items()
            if value is not None
        }


class ModelConfigRepository(Protocol):
    """配置解析器依赖的存储边界。

    resolver 只依赖这个协议，不直接依赖 SQLAlchemy。这样测试、Postgres 适配器
    和未来其它存储适配器可以复用同一套模型选择逻辑。
    """

    async def get_active_profile_by_alias(self, alias: str) -> ModelProfile | None: ...

    async def get_default_active_profile(self, *, scope: str = "global") -> ModelProfile | None: ...

    async def save_effective_snapshot(
        self, snapshot: EffectiveModelConfigSnapshot
    ) -> EffectiveModelConfigSnapshot: ...


class ModelConfigResolver:
    """把 DB、env、请求参数解析成一份最终模型配置。

    智能体图应该消费解析后的 config 或 snapshot id，不应该在执行过程中
    反复读取 env 或数据库里的 live 默认配置。
    """

    def __init__(self, repository: ModelConfigRepository, settings: Settings):
        self.repository = repository
        self.settings = settings

    async def resolve(
        self,
        *,
        selection: ModelSelection | None = None,
        requested_params: RequestedModelParams | None = None,
    ) -> EffectiveModelConfig:
        profile = await self._select_profile(selection)
        overrides = (requested_params or RequestedModelParams()).to_overrides()

        if profile is not None:
            # 数据库模型档案优先于 env；请求参数只能覆盖 RequestedModelParams
            # 明确列出的生成选项。
            source = "request_override" if selection and selection.model_alias else "db_default"
            params = {**profile.default_params, **overrides}
            credential = profile.credential
            config = self._build_config(
                source=source,
                provider=profile.provider,
                base_url=profile.base_url,
                model_name=profile.model_name,
                params=params,
                credential=credential,
                model_profile_id=profile.id,
                model_profile_version=profile.version,
            )
            return await self._save_snapshot(config)

        # env 是启动兜底：在后台管理或数据库 seed 创建默认模型前，服务也能跑通。
        params = {
            "temperature": self.settings.llm_temperature,
            "timeout_seconds": self.settings.llm_timeout_seconds,
            **overrides,
        }
        credential = None
        if self.settings.llm_api_key:
            credential = ProviderCredential(
                id=None,
                provider=self.settings.llm_provider,
                name="env",
                api_key=self.settings.llm_api_key,
            )
        config = self._build_config(
            source="env_fallback",
            provider=self.settings.llm_provider,
            base_url=self.settings.llm_base_url,
            model_name=self.settings.llm_model,
            params=params,
            credential=credential,
        )
        return await self._save_snapshot(config)

    async def _select_profile(self, selection: ModelSelection | None) -> ModelProfile | None:
        if selection and selection.model_alias:
            return await self.repository.get_active_profile_by_alias(selection.model_alias)
        return await self.repository.get_default_active_profile(scope="global")

    async def _save_snapshot(self, config: EffectiveModelConfig) -> EffectiveModelConfig:
        # 快照只保存模型元数据和凭证引用。密钥值只留在内存对象里，不复制到快照行。
        with tracer.start_as_current_span("llm.config_snapshot.save") as span:
            span.set_attribute("llm.config.source", config.source)
            span.set_attribute("llm.config.provider", config.provider)
            span.set_attribute("llm.config.model_name", config.model_name)
            span.set_attribute("llm.config.digest", config.digest)

            snapshot = await self.repository.save_effective_snapshot(
                EffectiveModelConfigSnapshot(
                    source=config.source,
                    model_profile_id=config.model_profile_id,
                    model_profile_version=config.model_profile_version,
                    provider=config.provider,
                    base_url=config.base_url,
                    model_name=config.model_name,
                    params=config.params,
                    credential_id=config.credential.id if config.credential else None,
                    config_digest=config.digest,
                )
            )
            span.set_attribute("llm.config.snapshot_id", str(snapshot.id) if snapshot.id else "")

        # EffectiveModelConfig 是 frozen dataclass，所以用 replace 返回一份带 snapshot_id 的新对象。
        return replace(config, snapshot_id=snapshot.id)

    def _build_config(
        self,
        *,
        source: str,
        provider: str,
        base_url: str,
        model_name: str,
        params: dict[str, object],
        credential: ProviderCredential | None,
        model_profile_id=None,
        model_profile_version: int | None = None,
    ) -> EffectiveModelConfig:
        # digest 覆盖不可变且不含密钥的配置形状，方便 resume/replay 时发现漂移。
        digest = self._digest(
            {
                "source": source,
                "model_profile_id": str(model_profile_id) if model_profile_id else None,
                "model_profile_version": model_profile_version,
                "provider": provider,
                "base_url": base_url,
                "model_name": model_name,
                "params": params,
                "credential_id": str(credential.id) if credential and credential.id else None,
            }
        )
        return EffectiveModelConfig(
            source=source,
            model_profile_id=model_profile_id,
            model_profile_version=model_profile_version,
            provider=provider,
            base_url=base_url,
            model_name=model_name,
            params=params,
            credential=credential,
            digest=digest,
        )

    @staticmethod
    def _digest(payload: dict[str, object]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
