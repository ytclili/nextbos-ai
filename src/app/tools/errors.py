class ToolError(Exception):
    pass


class ToolNotFoundError(ToolError):
    pass


class ToolPermissionDeniedError(ToolError):
    pass


class ToolTimeoutError(ToolError):
    pass
