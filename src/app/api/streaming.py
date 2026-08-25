import json
from typing import Any


def encode_sse_event(event: str, data: dict[str, Any]) -> str:
    """把业务事件编码成 SSE 文本。

    SSE 格式要求每个事件由 event/data 字段组成，并用空行结束。
    这里统一使用紧凑 JSON，减少 token 流式输出时的网络体积。
    """

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"
