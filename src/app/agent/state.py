from typing import Annotated, Any, TypedDict
from uuid import UUID

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.agent.options import ChatModelOptions


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    # LangMem SummarizationNode 默认会把压缩后的模型输入写到 summarized_messages。
    # 原始 messages 继续保留，respond 节点后续会优先读取 summarized_messages。
    summarized_messages: list[AnyMessage]

    # LangMem SummarizationNode 会把滚动摘要状态写到 context.running_summary。
    # 这里保留官方节点需要的上下文字段，不自己定义 summary 数据结构。
    context: dict[str, Any]

    user_id: str
    thread_id: str
    model_options: ChatModelOptions

    # 最后一次 LLM 调用使用的模型配置快照 id。
    # AgentService 会把它写入 assistant 消息，方便从聊天记录反查模型配置。
    llm_snapshot_id: UUID | None
