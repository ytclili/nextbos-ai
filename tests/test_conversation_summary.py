from uuid import UUID

from langmem.short_term import RunningSummary

from app.conversation.summary import (
    ExtractedConversationSummary,
    extract_running_summary,
)


def test_extract_running_summary_from_langmem_running_summary_object() -> None:
    """应该能从 LangMem RunningSummary 对象中提取可持久化摘要。"""

    message_id = "00000000-0000-0000-0000-000000000001"
    result = {
        "context": {
            "running_summary": RunningSummary(
                summary=" 用户说他的好朋友叫小亮。 ",
                summarized_message_ids={message_id, "message-2"},
                last_summarized_message_id=message_id,
            )
        }
    }

    summary = extract_running_summary(result)

    assert summary == ExtractedConversationSummary(
        summary="用户说他的好朋友叫小亮。",
        covered_through_message_id=UUID(message_id),
        message_count=2,
    )


def test_extract_running_summary_from_dict_shape() -> None:
    """如果 LangMem 输出被序列化成 dict，也应该能提取摘要。"""

    result = {
        "context": {
            "running_summary": {
                "summary": "用户喜欢粤菜。",
                "summarized_message_ids": ["message-1", "message-2", "message-3"],
                "last_summarized_message_id": "message-3",
            }
        }
    }

    summary = extract_running_summary(result)

    assert summary == ExtractedConversationSummary(
        summary="用户喜欢粤菜。",
        covered_through_message_id=None,
        message_count=3,
    )


def test_extract_running_summary_returns_none_when_summary_is_missing_or_blank() -> None:
    """没有 summary 或 summary 为空时，不应该保存无意义摘要。"""

    assert extract_running_summary({}) is None
    assert extract_running_summary({"context": {}}) is None
    assert (
        extract_running_summary(
            {
                "context": {
                    "running_summary": {
                        "summary": "   ",
                        "summarized_message_ids": ["message-1"],
                        "last_summarized_message_id": "message-1",
                    }
                }
            }
        )
        is None
    )


def test_extract_running_summary_ignores_invalid_summarized_message_ids() -> None:
    """非列表类 summarized_message_ids 不应该影响摘要文本提取。"""

    result = {
        "context": {
            "running_summary": {
                "summary": "用户在开发 nextbos-ai。",
                "summarized_message_ids": "message-1",
                "last_summarized_message_id": "",
            }
        }
    }

    summary = extract_running_summary(result)

    assert summary == ExtractedConversationSummary(
        summary="用户在开发 nextbos-ai。",
        covered_through_message_id=None,
        message_count=0,
    )
