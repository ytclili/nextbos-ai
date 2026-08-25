from dataclasses import dataclass, field
from uuid import UUID


@dataclass(frozen=True)
class ProviderCredential:
    """解析 DB 或 env 后得到的运行时凭证。

    API key 只允许存在于这个内存对象里，用完即走；持久化快照只能保存
    credential_id，不能复制密钥值。
    """

    id: UUID | None
    provider: str
    name: str
    api_key: str | None = None


@dataclass(frozen=True)
class ModelProfile:
    """数据库管理的模型档案。

    alias 是系统内部稳定使用的业务名，model_name 是真正传给 LangChain /
    上游供应商的模型名。
    """

    id: UUID
    alias: str
    provider: str
    display_name: str
    model_name: str
    base_url: str
    credential: ProviderCredential | None
    default_params: dict[str, object] = field(default_factory=dict)
    status: str = "active"
    is_default: bool = False
    scope: str = "global"
    scope_id: str | None = None
    version: int = 1


@dataclass(frozen=True)
class EffectiveModelConfig:
    """一次 agent run 或 chat 请求最终使用的模型配置。

    它是请求参数、数据库模型档案和 env 兜底值合并后的结果。digest 用来标识
    这份不含密钥的最终配置。
    """

    source: str
    provider: str
    base_url: str
    model_name: str
    params: dict[str, object]
    credential: ProviderCredential | None
    digest: str
    # 这次配置落库后的快照 id。assistant 消息会保存它，用来反查当时真实使用的模型配置。
    snapshot_id: UUID | None = None
    model_profile_id: UUID | None = None
    model_profile_version: int | None = None


@dataclass(frozen=True)
class EffectiveModelConfigSnapshot:
    """最终模型配置的不可变持久化快照。

    它保证可恢复的 agent run 不会因为后台默认模型变化而漂移。这里刻意只存
    credential_id，不存 API key。
    """

    source: str
    provider: str
    base_url: str
    model_name: str
    params: dict[str, object]
    config_digest: str
    # PostgreSQL 保存快照后生成的主键 id；创建前可以为空。
    id: UUID | None = None
    credential_id: UUID | None = None
    model_profile_id: UUID | None = None
    model_profile_version: int | None = None
