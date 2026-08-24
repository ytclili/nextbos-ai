from uuid import uuid4

import pytest

from app.core.config import Settings
from app.llm.chat_models import create_langchain_chat_model
from app.llm.config_resolver import ModelConfigResolver, ModelSelection, RequestedModelParams
from app.llm.models import ModelProfile, ProviderCredential
from app.persistence.postgres.models import LLMEffectiveConfigSnapshot, LLMModelProfile


class InMemoryModelRepository:
    """测试用内存仓储，避免单元测试依赖真实 PostgreSQL。"""

    def __init__(self, profile: ModelProfile | None = None):
        self.profile = profile
        self.saved_snapshot = None

    async def get_active_profile_by_alias(self, alias: str) -> ModelProfile | None:
        if self.profile and self.profile.alias == alias and self.profile.status == "active":
            return self.profile
        return None

    async def get_default_active_profile(self, *, scope: str = "global") -> ModelProfile | None:
        if self.profile and self.profile.is_default and self.profile.scope == scope:
            return self.profile
        return None

    async def save_effective_snapshot(self, snapshot):
        self.saved_snapshot = snapshot
        return snapshot


@pytest.mark.asyncio
async def test_resolver_uses_env_model_when_database_has_no_default_profile():
    settings = Settings(
        llm_provider="openai_compatible",
        llm_base_url="https://api.deepseek.com/v1",
        llm_model="deepseek-chat",
        llm_api_key="env-key",
        llm_temperature=0.2,
        llm_timeout_seconds=30,
    )
    repository = InMemoryModelRepository()

    config = await ModelConfigResolver(repository, settings).resolve()

    assert config.source == "env_fallback"
    assert config.provider == "openai_compatible"
    assert config.base_url == "https://api.deepseek.com/v1"
    assert config.model_name == "deepseek-chat"
    assert config.params == {"temperature": 0.2, "timeout_seconds": 30}
    assert config.credential.api_key == "env-key"
    assert config.digest
    assert repository.saved_snapshot is not None


@pytest.mark.asyncio
async def test_resolver_prefers_requested_database_profile_and_overrides_allowed_params():
    credential_id = uuid4()
    profile = ModelProfile(
        id=uuid4(),
        alias="default-chat",
        provider="openai_compatible",
        display_name="DeepSeek Chat",
        model_name="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        credential=ProviderCredential(
            id=credential_id,
            provider="openai_compatible",
            name="deepseek-prod",
            api_key="db-key",
        ),
        default_params={"temperature": 0.7, "max_tokens": 4096, "timeout_seconds": 60},
        status="active",
        is_default=True,
        scope="global",
        version=3,
    )
    repository = InMemoryModelRepository(profile)

    config = await ModelConfigResolver(repository, Settings()).resolve(
        selection=ModelSelection(model_alias="default-chat"),
        requested_params=RequestedModelParams(temperature=0.1, top_p=0.9),
    )

    assert config.source == "request_override"
    assert config.model_profile_id == profile.id
    assert config.model_profile_version == 3
    assert config.credential.id == credential_id
    assert config.params == {
        "temperature": 0.1,
        "max_tokens": 4096,
        "timeout_seconds": 60,
        "top_p": 0.9,
    }
    assert repository.saved_snapshot.config_digest == config.digest


def test_llm_model_profile_table_keeps_capabilities_out_of_first_version():
    columns = set(LLMModelProfile.__table__.columns.keys())

    assert {
        "id",
        "alias",
        "provider",
        "display_name",
        "model_name",
        "base_url",
        "credential_id",
        "default_params",
        "status",
        "is_default",
        "scope",
        "scope_id",
        "version",
    }.issubset(columns)
    assert "capabilities" not in columns


def test_effective_snapshot_stores_credential_reference_not_secret_value():
    columns = set(LLMEffectiveConfigSnapshot.__table__.columns.keys())

    assert "credential_id" in columns
    assert "api_key" not in columns
    assert "encrypted_api_key" not in columns


@pytest.mark.asyncio
async def test_chat_model_factory_builds_langchain_model_from_effective_config():
    captured_kwargs = {}

    class FakeChatModel:
        """测试用 LangChain 聊天模型，验证创建参数。"""

        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    config = await ModelConfigResolver(
        InMemoryModelRepository(),
        Settings(
            llm_provider="openai_compatible",
            llm_base_url="https://api.deepseek.com/v1",
            llm_model="deepseek-chat",
            llm_api_key="env-key",
            llm_temperature=0.2,
            llm_timeout_seconds=30,
        ),
    ).resolve(requested_params=RequestedModelParams(max_tokens=128, top_p=0.9))

    chat_model = create_langchain_chat_model(config, chat_model_factory=FakeChatModel)

    assert isinstance(chat_model, FakeChatModel)
    assert captured_kwargs == {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "env-key",
        "temperature": 0.2,
        "max_tokens": 128,
        "top_p": 0.9,
        "timeout": 30,
    }
