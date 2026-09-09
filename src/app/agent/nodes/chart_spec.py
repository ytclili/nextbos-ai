import json
from collections.abc import Awaitable, Callable
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from app.agent.schemas.chart import ChartPlan, ChartSpec, ChartSpecResult
from app.agent.state import AgentState

ChartSpecNode = Callable[[AgentState], Awaitable[dict[str, Any]]]


def create_chart_spec_node() -> ChartSpecNode:
    """创建图表配置节点。

    这个节点不调用 LLM，只把已校验的 ChartPlan 和真实 SQL rows 组装成 ECharts option。
    """

    async def chart_spec(state: AgentState) -> dict[str, Any]:
        """生成前端可渲染的 ECharts option JSON。"""

        result = _build_chart_spec_result(
            state.get("chart_plan_result") or {},
            state.get("sql_execution_result") or {},
        )
        output: dict[str, Any] = {"chart_spec_result": result.model_dump()}
        if result.chart is not None:
            output["messages"] = [_chart_spec_message(result.chart.option)]
        return output

    return chart_spec


def _build_chart_spec_result(
    chart_plan_result: dict[str, Any],
    sql_execution_result: dict[str, Any],
) -> ChartSpecResult:
    if sql_execution_result.get("status") != "succeeded":
        return ChartSpecResult(status="skipped", message="SQL 未成功执行，无法生成图表。")

    rows = sql_execution_result.get("rows") or []
    if not rows:
        return ChartSpecResult(status="skipped", message="SQL 结果为空，无法生成图表。")

    try:
        plan = ChartPlan.model_validate(chart_plan_result)
    except ValidationError:
        return ChartSpecResult(status="skipped", message="ChartPlan 不完整，无法生成图表。")

    available_fields = _available_fields(sql_execution_result)
    required_fields = _required_fields(plan)
    missing_fields = sorted(required_fields - available_fields)
    if missing_fields:
        return ChartSpecResult(
            status="skipped",
            message=f"ChartPlan 字段不存在于 SQL 结果中：{', '.join(missing_fields)}",
        )

    chart = (
        _build_pie_chart(plan, rows)
        if plan.chart_type == "pie"
        else _build_axis_chart(plan, rows)
    )
    return ChartSpecResult(status="succeeded", chart=chart)


def _available_fields(sql_execution_result: dict[str, Any]) -> set[str]:
    columns = sql_execution_result.get("columns") or []
    column_fields = {str(column["name"]) for column in columns if column.get("name")}
    row_fields = set().union(*(row.keys() for row in sql_execution_result.get("rows") or []))
    return column_fields | {str(field) for field in row_fields}


def _required_fields(plan: ChartPlan) -> set[str]:
    fields = {plan.x.field, plan.y.field}
    if plan.series is not None:
        fields.add(plan.series.field)
    return fields


def _build_axis_chart(plan: ChartPlan, rows: list[dict[str, Any]]) -> ChartSpec:
    x_label = plan.x.label or plan.x.field
    y_label = plan.y.label or plan.y.field
    option = {
        "title": {"text": plan.title},
        "tooltip": {"trigger": "axis"},
        "xAxis": {
            "type": "category",
            "name": x_label,
            "data": [_json_value(row.get(plan.x.field)) for row in rows],
        },
        "yAxis": {"type": "value", "name": y_label},
        "series": [
            {
                "type": plan.chart_type,
                "name": y_label,
                "data": [_json_value(row.get(plan.y.field)) for row in rows],
            }
        ],
    }
    return ChartSpec(
        title=plan.title,
        chart_type=plan.chart_type,
        option=option,
        source_fields=sorted(_required_fields(plan)),
        row_count=len(rows),
    )


def _build_pie_chart(plan: ChartPlan, rows: list[dict[str, Any]]) -> ChartSpec:
    y_label = plan.y.label or plan.y.field
    option = {
        "title": {"text": plan.title},
        "tooltip": {"trigger": "item"},
        "series": [
            {
                "type": "pie",
                "name": y_label,
                "data": [
                    {
                        "name": _json_value(row.get(plan.x.field)),
                        "value": _json_value(row.get(plan.y.field)),
                    }
                    for row in rows
                ],
            }
        ],
    }
    return ChartSpec(
        title=plan.title,
        chart_type=plan.chart_type,
        option=option,
        source_fields=sorted(_required_fields(plan)),
        row_count=len(rows),
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, date | datetime):
        return value.isoformat()
    return value


def _chart_spec_message(option: dict[str, Any]) -> AIMessage:
    content = (
        "前端 ECharts option：\n"
        "```json\n"
        f"{json.dumps(option, ensure_ascii=False, default=str, indent=2)}\n"
        "```"
    )
    return AIMessage(content=content)
