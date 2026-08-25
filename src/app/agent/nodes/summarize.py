from typing import Any

from langmem.short_term import SummarizationNode

from app.agent.state import AgentState

SUMMARY_MAX_TOKENS = 6000
SUMMARY_TRIGGER_TOKENS = 5000
SUMMARY_MAX_OUTPUT_TOKENS = 512


def create_summarize_node(summarization_model: Any | None):
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

    return SummarizationNode(
        model=summarization_model,
        max_tokens=SUMMARY_MAX_TOKENS,
        max_tokens_before_summary=SUMMARY_TRIGGER_TOKENS,
        max_summary_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
    )


def skip_summarization(state: AgentState) -> dict[str, list]:
    """没有 summary 模型时跳过总结。"""

    return {"summarized_messages": state.get("messages", [])}