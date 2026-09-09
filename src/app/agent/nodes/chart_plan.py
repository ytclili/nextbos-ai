import json
from collections.abc import Awaitable, Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from pydantic import ValidationError

from app.agent.model_runtime import AgentModelRuntime
from app.agent.schemas.chart import ChartPlan
from app.agent.state import AgentState

ChartPlanNode = Callable[[AgentState], Awaitable[dict[str, Any]]]
CHART_PLAN_PLACEHOLDER_MESSAGE = "已识别为图表请求，下一步生成 ChartPlan。"
CHART_PLAN_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "chart_plan.md"
MAX_CHART_PLAN_SAMPLE_ROWS = 20


def create_chart_plan_node(*, model_runtime: AgentModelRuntime | None = None) -> ChartPlanNode:
    """创建图表规划节点。

    这个节点只负责让 LLM 根据用户问题和 SQL 结果选择图表类型、标题和字段映射；
    真正的 ECharts option 由后续 chart_spec 节点基于真实 rows 确定性生成。
    """

    async def chart_plan(state: AgentState) -> dict[str, Any]:
        """调用 LLM 输出 ChartPlan，并写入 LangGraph state。"""

        if model_runtime is None:
            return {
                "chart_plan_result": {
                    "status": "pending",
                    "message": CHART_PLAN_PLACEHOLDER_MESSAGE,
                },
                "messages": [AIMessage(content=CHART_PLAN_PLACEHOLDER_MESSAGE)],
            }

        messages = state.get("summarized_messages") or state.get("messages", [])
        config = await model_runtime.resolve_config(state.get("model_options"))
        chat_model = model_runtime.create_chat_model(config)
        structured_model = chat_model.with_structured_output(ChartPlan)

        try:
            plan = ChartPlan.model_validate(
                await structured_model.ainvoke(_chart_plan_messages(state, messages))
            )
        except ValidationError:
            return {
                "chart_plan_result": {
                    "status": "failed",
                    "message": "ChartPlan 结构化输出校验失败。",
                },
                "llm_snapshot_id": config.snapshot_id,
            }

        return {
            "chart_plan_result": plan.model_dump(),
            "llm_snapshot_id": config.snapshot_id,
        }

    return chart_plan


@lru_cache(maxsize=1)
def load_chart_plan_prompt() -> str:
    """读取图表规划提示词。"""

    return CHART_PLAN_PROMPT_PATH.read_text(encoding="utf-8").strip()


def _chart_plan_messages(
    state: AgentState,
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """构造图表规划输入：保留对话上下文，并显式传入 SQL 结果摘要。"""

    payload = {
        "intent_decision": state.get("intent_decision") or {},
        "sql_execution_result": _compact_sql_execution_result(
            state.get("sql_execution_result") or {}
        ),
    }
    return [
        SystemMessage(content=load_chart_plan_prompt()),
        *messages,
        SystemMessage(
            content=(
                "当前需要生成 ChartPlan 的结构化输入是：\n"
                f"{json.dumps(payload, ensure_ascii=False, default=str)}"
            )
        ),
    ]


def _compact_sql_execution_result(execution_result: dict[str, Any]) -> dict[str, Any]:
    """压缩 SQL 执行结果，只给 LLM 判断图表所需的字段和少量样本。"""

    rows = execution_result.get("rows")
    if not isinstance(rows, list):
        rows = []

    return {
        "status": execution_result.get("status"),
        "columns": execution_result.get("columns") or [],
        "rows": rows[:MAX_CHART_PLAN_SAMPLE_ROWS],
        "row_count": execution_result.get("row_count", len(rows)),
        "truncated": execution_result.get("truncated", False)
        or len(rows) > MAX_CHART_PLAN_SAMPLE_ROWS,
    }
