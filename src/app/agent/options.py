from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatModelOptions:
    """一次 chat 请求传入的可选模型配置。

    这里放的是“调用方想怎么选模型”，不是最终模型配置。
    最终模型配置后面会由 ModelConfigResolver 结合 DB 和 env 解析出来。
    """

    model_alias: str | None = None
    model_params: dict[str, Any] | None = None
