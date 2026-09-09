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

    # 意图识别节点输出的结构化判断结果。
    # 这里保存 IntentDecision.model_dump() 后的可序列化快照，便于 Studio 展示、checkpoint 和回放。
    intent_decision: dict[str, Any]

    # 后端条件路由算出的下一跳名称，例如 direct_answer / wren_context / clarify。
    # 这个字段用于调试和回放，不把“下一跳怎么走”完全交给 LLM 自由决定。
    route_target: str

    # WrenAI 上下文查询结果。
    # 这里保存 WrenContextResult.model_dump() 后的可序列化快照，供后续 SQL 规划节点使用。
    wren_context: dict[str, Any]

    # WrenAI 上下文查询请求。
    # 这里保存 WrenContextRequest.model_dump() 后的可序列化快照，方便 Studio 调试和回放。
    wren_context_request: dict[str, Any]

    # SQL 规划节点输出的结构化计划。
    # 这里保存 SqlGenerationPlan.model_dump() 后的可序列化快照，供后续 dry-run 节点使用。
    sql_generation_plan: dict[str, Any]

    # Wren dry-plan / dry-run 节点输出的 SQL 校验结果。
    # 这里保存 SqlValidationResult.model_dump() 后的可序列化快照，防止后续节点误执行未通过 SQL。
    sql_validation_result: dict[str, Any]

    # SQL 受控执行节点输出的数据集快照。
    # 这里保存 SqlExecutionResult.model_dump() 后的可序列化结果，供最终回答、表格和图表节点使用。
    sql_execution_result: dict[str, Any]

    # 最后一次 LLM 调用使用的模型配置快照 id。
    # AgentService 会把它写入 assistant 消息，方便从聊天记录反查模型配置。
    llm_snapshot_id: UUID | None
