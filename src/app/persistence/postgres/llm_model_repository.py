from collections.abc import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.models import EffectiveModelConfigSnapshot, ModelProfile, ProviderCredential
from app.persistence.postgres.models import (
    LLMEffectiveConfigSnapshot,
    LLMModelProfile,
    LLMProviderCredential,
)


class PostgresLLMModelRepository:
    """LLM 模型配置的 PostgreSQL 适配器。

    这里负责把 ORM 行转换成 LLM 领域模型，并保存 effective config snapshot。
    resolver 只依赖协议，不直接知道这些 SQLAlchemy 细节。
    """

    def __init__(self, session: AsyncSession, decrypt_api_key: Callable[[str], str] | None = None):
        self.session = session
        self.decrypt_api_key = decrypt_api_key

    async def get_active_profile_by_alias(self, alias: str) -> ModelProfile | None:
        """按业务 alias 查找可用模型档案。"""

        statement = select(LLMModelProfile).where(
            LLMModelProfile.alias == alias,
            LLMModelProfile.status == "active",
        )
        row = await self.session.scalar(statement)
        if row is None:
            return None
        return await self._to_domain(row)

    async def get_default_active_profile(self, *, scope: str = "global") -> ModelProfile | None:
        """查找当前 scope 下的默认可用模型档案。"""

        statement = select(LLMModelProfile).where(
            LLMModelProfile.scope == scope,
            LLMModelProfile.status == "active",
            LLMModelProfile.is_default.is_(True),
        )
        row = await self.session.scalar(statement)
        if row is None:
            return None
        return await self._to_domain(row)

    async def save_effective_snapshot(
        self, snapshot: EffectiveModelConfigSnapshot
    ) -> EffectiveModelConfigSnapshot:
        """保存一次运行的最终模型配置快照。

        snapshot 只保存 credential_id 引用，不保存解密后的 API key。
        """

        row = LLMEffectiveConfigSnapshot(
            source=snapshot.source,
            model_profile_id=snapshot.model_profile_id,
            model_profile_version=snapshot.model_profile_version,
            provider=snapshot.provider,
            base_url=snapshot.base_url,
            model_name=snapshot.model_name,
            params=snapshot.params,
            credential_id=snapshot.credential_id,
            config_digest=snapshot.config_digest,
        )
        self.session.add(row)
        await self.session.commit()
        return snapshot

    async def _to_domain(self, row: LLMModelProfile) -> ModelProfile:
        """把 ORM 行转换成 LLM 领域模型。"""

        credential = None
        if row.credential_id:
            credential_row = await self.session.get(LLMProviderCredential, row.credential_id)
            if credential_row:
                credential = self._credential_to_domain(credential_row)

        return ModelProfile(
            id=row.id,
            alias=row.alias,
            provider=row.provider,
            display_name=row.display_name,
            model_name=row.model_name,
            base_url=row.base_url,
            credential=credential,
            default_params=row.default_params,
            status=row.status,
            is_default=row.is_default,
            scope=row.scope,
            scope_id=row.scope_id,
            version=row.version,
        )

    def _credential_to_domain(self, row: LLMProviderCredential) -> ProviderCredential:
        """把凭证行转换成运行时凭证对象。

        encrypted_api_key 只有在注入 decrypt_api_key 时才会解密；这让第一版可以
        先预留加密边界，而不是在仓储里写死某一种密钥实现。
        """

        api_key = None
        if row.encrypted_api_key and self.decrypt_api_key:
            api_key = self.decrypt_api_key(row.encrypted_api_key)
        return ProviderCredential(
            id=row.id,
            provider=row.provider,
            name=row.name,
            api_key=api_key,
        )
