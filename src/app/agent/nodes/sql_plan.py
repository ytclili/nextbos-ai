import json
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from app.agent.model_runtime import AgentModelRuntime
from app.agent.schemas.sql_plan import SqlGenerationPlan
from app.agent.state import AgentState

SqlPlanNode = Callable[[AgentState], Awaitable[dict[str, Any]]]
SQL_PLAN_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "sql_plan.md"


def create_sql_plan_node(*, model_runtime: AgentModelRuntime) -> SqlPlanNode:
    """创建 SQL 规划节点。

    这个节点只负责把 intent 和 WrenAI 上下文整理成结构化 SQL 计划；
    真正的 dry-plan、dry-run、权限控制和 SQL 执行都在后续节点完成。
    """

    async def sql_plan(state: AgentState) -> dict[str, Any]:
        """调用 LLM 输出 SqlGenerationPlan，并写入 LangGraph state。"""

        messages = state.get("summarized_messages") or state.get("messages", [])
        question = _current_question(state, messages)

        config = await model_runtime.resolve_config(state.get("model_options"))
        chat_model = model_runtime.create_chat_model(config)
        structured_model = chat_model.with_structured_output(SqlGenerationPlan)
        try:
            plan = SqlGenerationPlan.model_validate(
                await structured_model.ainvoke(_sql_plan_messages(state, messages, question))
            )
        except ValidationError:
            plan = _fallback_sql_generation_plan(question)

        return {
            "sql_generation_plan": plan.model_dump(),
            "llm_snapshot_id": config.snapshot_id,
        }

    return sql_plan


@lru_cache(maxsize=1)
def load_sql_plan_prompt() -> str:
    """读取 SQL 规划提示词。"""

    return SQL_PLAN_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _sql_plan_messages(
    state: AgentState,
    messages: list[BaseMessage],
    question: str,
) -> list[BaseMessage]:
    """构造 SQL 规划输入：保留对话上下文，并明确传入结构化业务上下文。"""

    payload = {
        "current_question": question,
        "intent_decision": state.get("intent_decision") or {},
        "wren_context_request": state.get("wren_context_request") or {},
        "wren_context": state.get("wren_context") or {},
    }
    return [
        SystemMessage(content=load_sql_plan_prompt()),
        *messages,
        SystemMessage(
            content=(
                "当前需要生成 SQL 计划的结构化输入是：\n"
                f"{json.dumps(payload, ensure_ascii=False, default=str)}"
            )
        ),
    ]


def _current_question(state: AgentState, messages: list[BaseMessage]) -> str:
    """优先使用 intent 改写问题，其次使用 Wren 请求，最后回退到最近用户消息。"""

    decision = state.get("intent_decision") or {}
    if decision.get("question_rewrite"):
        return str(decision["question_rewrite"])

    request = state.get("wren_context_request") or {}
    if request.get("question"):
        return str(request["question"])

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _fallback_sql_generation_plan(question: str) -> SqlGenerationPlan:
    """LLM 结构化输出不合法时，返回不可执行的安全兜底计划。"""

    return SqlGenerationPlan(
        status="cannot_plan",
        query_intent="lookup",
        question=question or "当前业务问题",
        planning_summary="SQL 规划结构化输出校验失败，已停止进入 dry-run。",
        clarification_question="当前业务上下文不足以生成可靠 SQL，请补充更明确的查询条件。",
        confidence=0,
    )
