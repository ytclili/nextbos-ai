from langchain_core.tools import BaseTool

from app.tools.errors import ToolPermissionDeniedError


class ToolPolicy:
    def check(self, tool: BaseTool, *, allowed_tools: set[str] | None = None) -> None:
        if allowed_tools is not None and tool.name not in allowed_tools:
            raise ToolPermissionDeniedError(f"tool is not allowed: {tool.name}")
