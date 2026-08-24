class LLMError(Exception):
    """LLM 调用或响应解析失败。"""


class LLMConfigurationError(LLMError):
    """LLM 配置缺失或不受支持。"""
