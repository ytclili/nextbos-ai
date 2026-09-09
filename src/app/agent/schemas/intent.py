from typing import Literal

from pydantic import BaseModel, Field

IntentType = Literal[
    "direct_answer",
    "business_query",
    "business_analysis",
    "chart_request",
    "report_request",
    "clarification",
]
OutputType = Literal["text", "table", "chart", "report"]
RouteTarget = Literal["direct_answer", "wren_context", "clarify"]


class IntentFilter(BaseModel):
    """用户明确给出的单个过滤条件。"""

    field: str = Field(description="过滤字段或业务概念，例如区域、渠道、商品。")
    value: str = Field(description="过滤值，例如华东、小程序、某个商品名。")


class IntentDecision(BaseModel):
    """用户问题的结构化意图识别结果。

    这个对象只负责判断用户想做什么、是否需要业务数据、以及后续路由所需的关键信号。
    它不生成 SQL，也不承担数据库安全边界；真正的查询安全由后续 Wren 校验和 SQL 执行层负责。
    """

    intent_type: IntentType = Field(
        description="问题大类，例如直接回答、业务查询、业务分析、图表请求、报告请求或澄清。",
    )
    needs_business_data: bool = Field(
        description="是否需要查询业务数据。这里只作为路由信号，不作为 SQL 安全边界。",
    )
    output_type: OutputType = Field(
        description="用户期望的输出形态，例如文本、表格、图表或完整报告。",
    )
    question_rewrite: str = Field(
        min_length=1,
        description="把用户原始问题改写成清晰、完整、适合后续查询或分析的问题。",
    )
    metrics: list[str] = Field(
        description="用户关心的业务指标，例如销售额、订单数、退款率、毛利率。",
    )
    dimensions: list[str] = Field(
        description="用户要求分组、对比或下钻的维度，例如区域、门店、商品、客户、时间。",
    )
    filters: list[IntentFilter] = Field(
        description="用户明确给出的过滤条件列表，例如区域=华东、渠道=小程序。",
    )
    time_range: str | None = Field(
        description="用户明确或可推断的时间范围，例如本月、上周、2026-08。",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="模型对意图判断的置信度，范围 0 到 1。",
    )
    missing_slots: list[str] = Field(
        description="继续查询前缺失的关键信息，例如 time_range、metric、dimension。",
    )
    clarification_question: str | None = Field(
        description="需要用户补充信息时，用一句话提出澄清问题。",
    )
