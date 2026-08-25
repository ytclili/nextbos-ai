from dataclasses import dataclass
from typing import Any

from langmem.short_term import SummarizationNode

from app.agent.state import AgentState

DEFAULT_SUMMARY_MAX_TOKENS = 8000
DEFAULT_SUMMARY_TRIGGER_TOKENS = 6000
DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS = 800


@dataclass(frozen=True)
class SummaryOptions:
    """LangMem summary 节点的上下文压缩参数。"""

    max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS
    trigger_tokens: int = DEFAULT_SUMMARY_TRIGGER_TOKENS
    max_output_tokens: int = DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS


def create_summarize_node(
    summarization_model: Any | None,
    *,
    options: SummaryOptions | None = None,
):
    """创建官方 LangMem summary 节点。

    summarization_model 为空时返回 no-op 节点，方便测试和 fallback graph 继续运行。
    真正运行时由 runtime 传入当前模型创建出来的 ChatModel。

    这里使用 SummarizationNode 默认输出字段：
    summarized_messages。

    也就是说：
    - 原始 messages 不被覆盖；
    - 压缩后的模型输入写到 summarized_messages；
    - respond 节点后续优先读取 summarized_messages。
    """

    if summarization_model is None:
        return skip_summarization

    options = options or SummaryOptions()
    return SummarizationNode(
        model=summarization_model,
        max_tokens=options.max_tokens,
        max_tokens_before_summary=options.trigger_tokens,
        max_summary_tokens=options.max_output_tokens,
    )


def skip_summarization(state: AgentState) -> dict[str, list]:
    """没有 summary 模型时跳过总结。"""

    return {"summarized_messages": state.get("messages", [])}
