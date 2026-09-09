import pytest
from pydantic import ValidationError

from app.agent.schemas.chart import ChartEncoding, ChartPlan


def test_chart_plan_accepts_basic_bar_chart() -> None:
    """ChartPlan 只描述图表意图，不直接塞真实数据。"""

    plan = ChartPlan(
        chart_type="bar",
        title="华东销售额趋势",
        x=ChartEncoding(field="order_date", label="日期"),
        y=ChartEncoding(field="sales_amount", label="销售额"),
        series=None,
        description="按日期展示华东销售额变化。",
    )

    assert plan.chart_type == "bar"
    assert plan.x.field == "order_date"
    assert plan.y.field == "sales_amount"


def test_chart_plan_rejects_pie_chart_without_category_field() -> None:
    """饼图必须有分类字段，避免生成不可渲染的图表计划。"""

    with pytest.raises(ValidationError, match="饼图必须使用分类字段"):
        ChartPlan(
            chart_type="pie",
            title="销售额占比",
            x=None,
            y=ChartEncoding(field="sales_amount", label="销售额"),
            series=None,
            description="展示销售额占比。",
        )


def test_chart_plan_rejects_duplicate_axis_fields() -> None:
    """横轴和纵轴不能指向同一个字段。"""

    with pytest.raises(ValidationError, match="横轴和纵轴不能使用同一个字段"):
        ChartPlan(
            chart_type="bar",
            title="销售额",
            x=ChartEncoding(field="sales_amount", label="销售额"),
            y=ChartEncoding(field="sales_amount", label="销售额"),
            series=None,
            description="展示销售额。",
        )
