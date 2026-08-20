"""Agent 领域 Span 定义（预留）。

这里只放少量业务边界 Span，例如 chat.request、invoke_workflow、
context.load、execute_tool、save.assistant.message。
FastAPI、HTTPX、Redis、SQLAlchemy 等基础设施 Span 优先由官方自动埋点生成。
"""

# TODO: 添加领域 Span 工厂/上下文管理器，并统一属性、状态和错误字段。
