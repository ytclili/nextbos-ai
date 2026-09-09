from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from app.agent.model_runtime import AgentModelRuntime
from app.agent.schemas.intent import IntentDecision, RouteTarget
from app.agent.state import AgentState
from app.core.tracing import get_tracer

tracer = get_tracer(__name__)

IntentNode = Callable[[AgentState], Awaitable[dict[str, Any]]]
INTENT_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "intent.md"


def create_intent_node(*, model_runtime: AgentModelRuntime) -> IntentNode:
    """创建意图识别节点。

    这个节点只负责识别用户问题的结构化意图，并计算后续路由目标。
    真正的 SQL 生成、Wren 上下文查询和数据库安全检查都在后续节点完成。
    """

    async def intent(state: AgentState) -> dict[str, Any]:
        """调用 LLM 输出 IntentDecision，并保存确定性的 route_target。"""

        with tracer.start_as_current_span("agent.node.intent") as span:
            messages = state.get("summarized_messages") or state.get("messages", [])
            question = _latest_human_question(messages)
            span.set_attribute("agent.node.name", "intent")
            span.set_attribute("agent.thread_id", state.get("thread_id", ""))
            span.set_attribute("agent.user_id", state.get("user_id", ""))
            span.set_attribute("agent.intent.question_length", len(question))

            config = await model_runtime.resolve_config(state.get("model_options"))
            chat_model = model_runtime.create_chat_model(config)
            structured_model = chat_model.with_structured_output(IntentDecision)
            decision = IntentDecision.model_validate(
                await structured_model.ainvoke(_intent_messages(messages, question))
            )
            route_target = route_intent_decision(decision)

            span.set_attribute("agent.intent.type", decision.intent_type)
            span.set_attribute("agent.intent.needs_business_data", decision.needs_business_data)
            span.set_attribute("agent.intent.route_target", route_target)
            return {
                "intent_decision": decision.model_dump(),
                "route_target": route_target,
                "llm_snapshot_id": config.snapshot_id,
            }

    return intent


def route_intent_decision(decision: IntentDecision) -> RouteTarget:
    """用后端确定性规则选择下一跳，不把图路由完全交给 LLM。"""

    if decision.intent_type == "clarification" or decision.missing_slots:
        return "clarify"
    if decision.needs_business_data:
        return "wren_context"
    return "direct_answer"


@lru_cache(maxsize=1)
def load_intent_prompt() -> str:
    """读取意图识别提示词。"""

    return INTENT_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _intent_messages(messages: list[BaseMessage], question: str) -> list[BaseMessage]:
    """构造意图识别输入：保留上下文，同时明确标注当前问题。"""

    return [
        SystemMessage(content=load_intent_prompt()),
        *messages,
        SystemMessage(content=f"当前需要识别意图的用户问题是：\n{question}"),
    ]


def _latest_human_question(messages: list[BaseMessage]) -> str:
    """读取最近一条用户消息，作为本轮意图识别对象。"""

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""
