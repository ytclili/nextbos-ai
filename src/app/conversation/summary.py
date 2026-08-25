from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class ExtractedConversationSummary:
    """从 LangGraph result 中提取出的可持久化会话摘要。

    summary 是滚动摘要文本；
    covered_through_message_id 表示这份摘要覆盖到哪条 PostgreSQL message；
    message_count 是 LangMem 标记为已总结的消息数量。

    注意：
    LangMem 的 message id 不一定总是 PostgreSQL UUID。
    只有它能解析成 UUID 时，才写入 covered_through_message_id。
    """

    summary: str
    covered_through_message_id: UUID | None = None
    message_count: int = 0


def extract_running_summary(result: Mapping[str, Any]) -> ExtractedConversationSummary | None:
    """从 LangGraph 执行结果里提取 LangMem rolling summary。

    LangMem SummarizationNode 的输出形状通常是：

    {
        "context": {
            "running_summary": RunningSummary(...)
        }
    }

    RunningSummary 里包含：
    - summary：最新滚动摘要文本；
    - summarized_message_ids：已经被摘要覆盖的消息 id；
    - last_summarized_message_id：最后一条被摘要覆盖的消息 id。

    这里单独封装，是为了避免 AgentService 直接依赖 LangMem 对象细节。
    """

    context = result.get("context")
    if not isinstance(context, Mapping):
        return None

    running_summary = context.get("running_summary")
    if running_summary is None:
        return None

    summary = _get_value(running_summary, "summary")
    if not isinstance(summary, str) or not summary.strip():
        return None

    summarized_message_ids = _get_value(running_summary, "summarized_message_ids")
    last_summarized_message_id = _get_value(
        running_summary,
        "last_summarized_message_id",
    )

    return ExtractedConversationSummary(
        summary=summary.strip(),
        covered_through_message_id=_parse_uuid(last_summarized_message_id),
        message_count=_count_summarized_messages(summarized_message_ids),
    )


def _get_value(source: object, key: str) -> object:
    """兼容 LangMem dataclass 对象和 dict 形状。"""

    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _parse_uuid(value: object) -> UUID | None:
    """只有 message id 是 UUID 字符串时，才转成 PostgreSQL message id。"""

    if not isinstance(value, str) or not value:
        return None

    try:
        return UUID(value)
    except ValueError:
        return None


def _count_summarized_messages(value: object) -> int:
    """统计 LangMem 已经纳入 rolling summary 的消息数量。"""

    if isinstance(value, set | list | tuple):
        return len(value)
    return 0