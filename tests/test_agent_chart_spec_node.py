from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage

from app.agent.nodes.chart_spec import create_chart_spec_node


@pytest.mark.asyncio
async def test_chart_spec_node_builds_bar_echarts_option_from_real_rows() -> None:
    """图表配置必须由后端用真实 SQL rows 填充。"""

    node = create_chart_spec_node()

    result = await node(
        {
            "chart_plan_result": {
                "chart_type": "bar",
                "title": "华东销售额趋势",
                "x": {"field": "order_date", "label": "日期"},
                "y": {"field": "sales_amount", "label": "销售额"},
                "series": None,
                "description": "按日期展示销售额。",
            },
            "sql_execution_result": {
                "status": "succeeded",
                "columns": [
                    {"name": "order_date", "data_type": "date"},
                    {"name": "sales_amount", "data_type": "decimal"},
                ],
                "rows": [
                    {"order_date": "2026-09-01", "sales_amount": Decimal("100.50")},
                    {"order_date": "2026-09-02", "sales_amount": Decimal("200.00")},
                ],
                "row_count": 2,
            },
        }
    )

    chart_result = result["chart_spec_result"]
    assert chart_result["status"] == "succeeded"
    assert chart_result["chart"]["type"] == "echarts"
    assert chart_result["chart"]["option"]["xAxis"]["data"] == ["2026-09-01", "2026-09-02"]
    assert chart_result["chart"]["option"]["series"] == [
        {"type": "bar", "name": "销售额", "data": [100.5, 200.0]}
    ]
    assert isinstance(result["messages"][-1], AIMessage)
    assert "前端 ECharts option" in result["messages"][-1].content
    assert '"type": "bar"' in result["messages"][-1].content
    assert "100.5" in result["messages"][-1].content
    assert "200.0" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_chart_spec_node_builds_pie_echarts_option_from_real_rows() -> None:
    """饼图用分类字段作为 name，用数值字段作为 value。"""

    node = create_chart_spec_node()

    result = await node(
        {
            "chart_plan_result": {
                "chart_type": "pie",
                "title": "区域销售额占比",
                "x": {"field": "region", "label": "区域"},
                "y": {"field": "sales_amount", "label": "销售额"},
                "series": None,
                "description": "按区域展示销售额占比。",
            },
            "sql_execution_result": {
                "status": "succeeded",
                "columns": [
                    {"name": "region", "data_type": "varchar"},
                    {"name": "sales_amount", "data_type": "decimal"},
                ],
                "rows": [
                    {"region": "华东", "sales_amount": "100"},
                    {"region": "华南", "sales_amount": "80"},
                ],
                "row_count": 2,
            },
        }
    )

    chart_result = result["chart_spec_result"]
    assert chart_result["status"] == "succeeded"
    assert chart_result["chart"]["option"]["series"][0]["type"] == "pie"
    assert chart_result["chart"]["option"]["series"][0]["data"] == [
        {"name": "华东", "value": "100"},
        {"name": "华南", "value": "80"},
    ]


@pytest.mark.asyncio
async def test_chart_spec_node_skips_when_plan_field_not_in_sql_result() -> None:
    """ChartPlan 字段必须来自 SQL 返回结果。"""

    node = create_chart_spec_node()

    result = await node(
        {
            "chart_plan_result": {
                "chart_type": "bar",
                "title": "销售额",
                "x": {"field": "missing_date", "label": "日期"},
                "y": {"field": "sales_amount", "label": "销售额"},
                "series": None,
                "description": "展示销售额。",
            },
            "sql_execution_result": {
                "status": "succeeded",
                "columns": [{"name": "sales_amount", "data_type": "decimal"}],
                "rows": [{"sales_amount": "100"}],
                "row_count": 1,
            },
        }
    )

    assert result["chart_spec_result"]["status"] == "skipped"
    assert "missing_date" in result["chart_spec_result"]["message"]
