class AppError(Exception):
    """Base application error."""


class InfrastructureUnavailableError(AppError):
    """A required Redis/PostgreSQL dependency is unavailable."""
