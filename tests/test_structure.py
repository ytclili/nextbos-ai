from app.agent.graph import graph
from app.core.config import get_settings
from app.skills.loader import SkillLoader
from app.tools.registry import get_builtin_tool_names


def test_graph_is_compiled():
    assert graph is not None


def test_builtin_tool_registry_contains_builtin_tools():
    assert get_builtin_tool_names() == ["health_check", "manage_memory", "search_memory"]


def test_unknown_skill_returns_none(tmp_path):
    assert SkillLoader(tmp_path).load("missing") is None


def test_settings_have_infrastructure_defaults():
    settings = get_settings()
    assert settings.redis_url.startswith("redis://")
    assert settings.postgres_dsn.startswith("postgresql+asyncpg://")
