from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

ChartType = Literal["bar", "line", "pie"]
ChartSpecStatus = Literal["succeeded", "skipped", "failed"]


class ChartEncoding(BaseModel):
    """图表中的一个字段映射，例如横轴、纵轴或系列字段。"""

    field: str = Field(min_length=1, description="SQL 查询结果中的真实字段名。")
    label: str | None = Field(default=None, description="展示给用户看的字段名称。")


class ChartPlan(BaseModel):
    """图表规划节点的结构化输出。

    这个对象只描述图表类型和字段映射，不直接保存真实数据。
    真实 data 必须由后端根据 sql_execution_result 填充，避免 LLM 编造数据。
    """

    chart_type: ChartType = Field(description="图表类型，由模型根据用户问题和查询结果选择。")
    title: str = Field(min_length=1, description="图表标题。")
    x: ChartEncoding | None = Field(default=None, description="横轴或饼图分类字段。")
    y: ChartEncoding | None = Field(default=None, description="纵轴或饼图数值字段。")
    series: ChartEncoding | None = Field(default=None, description="可选的系列分组字段。")
    description: str = Field(description="一句话说明这个图表回答什么问题。")

    @model_validator(mode="after")
    def validate_chart_contract(self) -> Self:
        """约束图表计划，避免生成后端无法渲染的配置。"""

        if self.x is None or self.y is None:
            if self.chart_type == "pie":
                raise ValueError("饼图必须使用分类字段和数值字段")
            raise ValueError("柱状图和折线图必须包含横轴和纵轴字段")

        if self.x.field == self.y.field:
            raise ValueError("横轴和纵轴不能使用同一个字段")

        return self


class ChartSpec(BaseModel):
    """后端生成的图表渲染配置。

    这里保存的是 ECharts option JSON，不是前端可执行 JS 代码。
    option 中的数据必须来自 SQL 执行结果，不能由 LLM 直接编造。
    """

    type: Literal["echarts"] = Field(default="echarts", description="图表渲染类型。")
    title: str = Field(min_length=1, description="图表标题。")
    chart_type: ChartType = Field(description="ECharts series 使用的图表类型。")
    option: dict[str, Any] = Field(description="可直接交给前端 ECharts 渲染的 option JSON。")
    source_fields: list[str] = Field(description="图表使用的 SQL 结果字段。")
    row_count: int = Field(ge=0, description="图表使用的数据行数。")


class ChartSpecResult(BaseModel):
    """图表配置生成节点的结果快照。"""

    status: ChartSpecStatus = Field(description="图表配置生成状态。")
    chart: ChartSpec | None = Field(default=None, description="生成成功时的图表配置。")
    message: str | None = Field(default=None, description="跳过或失败时的原因。")

    @model_validator(mode="after")
    def validate_status_contract(self) -> Self:
        """约束状态和图表配置的关系，避免前端拿到半成品。"""

        if self.status == "succeeded":
            if self.chart is None:
                raise ValueError("succeeded 状态必须包含 chart")
            return self

        if self.chart is not None:
            raise ValueError("未成功生成图表时不能包含 chart")
        if not self.message:
            raise ValueError("跳过或失败时必须包含原因")
        return self
