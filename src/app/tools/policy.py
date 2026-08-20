from app.tools.errors import ToolPermissionDeniedError
from app.tools.registry import ToolSpec


class ToolPolicy:
    def check(self, tool: ToolSpec, *, allowed_tools: set[str] | None = None) -> None:
        if allowed_tools is not None and tool.name not in allowed_tools:
            raise ToolPermissionDeniedError(f"tool is not allowed: {tool.name}")
