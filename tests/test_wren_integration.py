from pathlib import Path

import pytest

from app.integrations.business.sql_executor import LangChainToolSqlExecutionClient
from app.integrations.business.wren import (
    WrenBusinessRule,
    WrenContextError,
    WrenContextFilter,
    WrenContextRequest,
    WrenContextResult,
    WrenField,
    WrenJoin,
    WrenLangChainContextClient,
    WrenMetric,
    WrenModel,
    WrenProjectConfig,
    WrenSqlExample,
)


def test_wren_context_request_serializes_intent_inputs() -> None:
    """WrenAI 上下文请求应该能承接 intent 节点输出的业务条件。"""

    request = WrenContextRequest(
        question="分析本月华东区域销售额下降原因",
        tenant_id="tenant-1",
        user_id="user-1",
        metrics=["销售额"],
        dimensions=["区域"],
        filters=[WrenContextFilter(field="区域", value="华东")],
        time_range="本月",
    )

    assert request.model_dump() == {
        "question": "分析本月华东区域销售额下降原因",
        "tenant_id": "tenant-1",
        "user_id": "user-1",
        "metrics": ["销售额"],
        "dimensions": ["区域"],
        "filters": [{"field": "区域", "value": "华东"}],
        "time_range": "本月",
    }


def test_wren_context_result_defaults_to_empty_context_snapshot() -> None:
    """没有真实 WrenAI 返回时，默认上下文快照应该保持结构稳定。"""

    result = WrenContextResult()

    assert result.model_dump() == {
        "models": [],
        "fields": [],
        "metrics": [],
        "joins": [],
        "business_rules": [],
        "historical_sql": [],
        "raw": {},
    }


def test_wren_context_result_serializes_standardized_context() -> None:
    """标准化后的 WrenAI 上下文应该能被后续 SQL 规划节点直接读取。"""

    result = WrenContextResult(
        models=[WrenModel(name="orders", description="订单模型")],
        fields=[
            WrenField(
                model="orders",
                name="amount",
                data_type="decimal",
                description="订单金额",
            )
        ],
        metrics=[
            WrenMetric(
                name="销售额",
                expression="sum(orders.amount)",
                description="订单金额合计",
            )
        ],
        joins=[
            WrenJoin(
                left_model="orders",
                right_model="stores",
                condition="orders.store_id = stores.id",
                description="订单关联门店",
            )
        ],
        business_rules=[WrenBusinessRule(name="有效订单", description="排除已取消订单")],
        historical_sql=[
            WrenSqlExample(
                question="查询本月销售额",
                sql="select sum(amount) from orders",
                note="示例 SQL 只用于参考",
            )
        ],
        raw={"query_id": "wren-query-1"},
    )

    dumped = result.model_dump()

    assert dumped["models"][0]["name"] == "orders"
    assert dumped["fields"][0]["name"] == "amount"
    assert dumped["metrics"][0]["name"] == "销售额"
    assert dumped["joins"][0]["condition"] == "orders.store_id = stores.id"
    assert dumped["business_rules"][0]["name"] == "有效订单"
    assert dumped["historical_sql"][0]["question"] == "查询本月销售额"
    assert dumped["raw"] == {"query_id": "wren-query-1"}


def test_wren_langchain_context_client_reads_official_tools() -> None:
    """Wren 适配层应该优先通过官方 LangChain tools 读取上下文。"""

    client = WrenLangChainContextClient(
        WrenProjectConfig(project_path=Path("/tmp/wren-project")),
        toolkit=FakeWrenToolkit(
            [
                FakeWrenTool(
                    "wren_list_models",
                    {
                        "ok": True,
                        "data": {"models": [{"name": "orders", "description": "订单模型"}]},
                    },
                ),
                FakeWrenTool(
                    "wren_fetch_context",
                    {
                        "ok": True,
                        "data": {
                            "results": [
                                {
                                    "item_type": "column",
                                    "model": "orders",
                                    "name": "amount",
                                    "data_type": "decimal",
                                    "summary": "订单金额",
                                },
                                {
                                    "item_type": "relationship",
                                    "left_model": "orders",
                                    "right_model": "stores",
                                    "condition": "orders.store_id = stores.id",
                                    "summary": "订单关联门店",
                                },
                            ]
                        },
                    },
                ),
                FakeWrenTool(
                    "wren_recall_queries",
                    {
                        "ok": True,
                        "data": {
                            "results": [
                                {
                                    "nl": "查询本月销售额",
                                    "sql": "select sum(amount) from orders",
                                }
                            ]
                        },
                    },
                ),
            ]
        ),
    )

    result = client.query_context_sync(
        WrenContextRequest(question="分析本月华东区域销售额下降原因")
    )

    assert result.models == [WrenModel(name="orders", description="订单模型")]
    assert result.fields == [
        WrenField(
            model="orders",
            name="amount",
            data_type="decimal",
            description="订单金额",
        )
    ]
    assert result.joins == [
        WrenJoin(
            left_model="orders",
            right_model="stores",
            condition="orders.store_id = stores.id",
            description="订单关联门店",
        )
    ]
    assert result.historical_sql == [
        WrenSqlExample(question="查询本月销售额", sql="select sum(amount) from orders")
    ]


def test_wren_langchain_context_client_supports_lightweight_runtime_tools() -> None:
    """轻量安装没有 memory 工具时，适配层仍应能读取模型列表。"""

    client = WrenLangChainContextClient(
        WrenProjectConfig(project_path=Path("/tmp/wren-project")),
        toolkit=FakeWrenToolkit(
            [
                FakeWrenTool(
                    "wren_list_models",
                    {
                        "ok": True,
                        "data": {"models": [{"name": "orders", "description": "订单模型"}]},
                    },
                )
            ]
        ),
    )

    result = client.query_context_sync(WrenContextRequest(question="查询本月销售额"))

    assert result.models == [WrenModel(name="orders", description="订单模型")]
    assert result.fields == []
    assert result.historical_sql == []


def test_wren_langchain_context_client_validates_sql_with_dry_plan() -> None:
    """Wren 适配层应该通过官方 wren_dry_plan 生成可执行方言 SQL 快照。"""

    dry_plan_tool = FakeWrenTool(
        "wren_dry_plan",
        {
            "ok": True,
            "content": "dry plan ok",
            "data": {"dialect_sql": "select sum(paid_amount) as sales_amount from orders"},
            "warnings": [],
        },
    )
    client = WrenLangChainContextClient(
        WrenProjectConfig(project_path=Path("/tmp/wren-project")),
        toolkit=FakeWrenToolkit([dry_plan_tool]),
    )

    result = client.validate_sql_sync(
        candidate_name="primary",
        sql="select sum(paid_amount) as sales_amount from orders",
    )

    assert dry_plan_tool.calls == [
        {"sql": "select sum(paid_amount) as sales_amount from orders"}
    ]
    assert result.status == "passed"
    assert result.validated_sql == "select sum(paid_amount) as sales_amount from orders"
    assert result.issues == []
    assert result.dry_plan["content"] == "dry plan ok"


def test_wren_langchain_context_client_returns_repair_result_when_dry_plan_fails() -> None:
    """Wren dry-plan 失败时，适配层应该返回可供 SQL 修复节点使用的错误快照。"""

    client = WrenLangChainContextClient(
        WrenProjectConfig(project_path=Path("/tmp/wren-project")),
        toolkit=FakeWrenToolkit(
            [
                FakeWrenTool(
                    "wren_dry_plan",
                    {
                        "ok": False,
                        "content": "column not found",
                        "error": {
                            "code": "COLUMN_NOT_FOUND",
                            "phase": "DRY_PLAN",
                            "message": "字段 paid_amount 不存在",
                            "metadata": {},
                        },
                    },
                )
            ]
        ),
    )

    result = client.validate_sql_sync(
        candidate_name="primary",
        sql="select paid_amount from orders",
    )

    assert result.status == "needs_repair"
    assert result.validated_sql is None
    assert result.issues[0].severity == "error"
    assert result.issues[0].message == "字段 paid_amount 不存在"


def test_wren_langchain_context_client_marks_failed_after_max_repair_attempts() -> None:
    """达到最大修复次数后，dry-plan 失败应该进入 failed。"""

    client = WrenLangChainContextClient(
        WrenProjectConfig(project_path=Path("/tmp/wren-project")),
        toolkit=FakeWrenToolkit([]),
    )

    result = client.validate_sql_sync(
        candidate_name="primary",
        sql="select paid_amount from orders",
        repair_attempt=2,
        max_repair_attempts=2,
    )

    assert result.status == "failed"
    assert result.issues[0].message == "Wren tool wren_dry_plan 不存在"


def test_wren_langchain_context_client_creates_sql_execution_client() -> None:
    """Wren 适配层应该能把官方 wren_query tool 包装成 SQL 执行客户端。"""

    query_tool = FakeWrenTool(
        "wren_query",
        {
            "ok": True,
            "data": {
                "columns": ["goods_id", "goods_name"],
                "rows": [{"goods_id": 1, "goods_name": "苹果"}],
            },
        },
    )
    client = WrenLangChainContextClient(
        WrenProjectConfig(project_path=Path("/tmp/wren-project")),
        toolkit=FakeWrenToolkit([query_tool]),
    )

    execution_client = client.create_sql_execution_client()
    result = execution_client.execute_sql_sync(sql="select id as goods_id from wms_items")

    assert isinstance(execution_client, LangChainToolSqlExecutionClient)
    assert query_tool.calls == [{"sql": "select id as goods_id from wms_items", "limit": 100}]
    assert result.status == "succeeded"
    assert result.rows == [{"goods_id": 1, "goods_name": "苹果"}]


def test_wren_langchain_context_client_requires_query_tool_for_execution() -> None:
    """没有 wren_query 时，不能创建 SQL 执行客户端。"""

    client = WrenLangChainContextClient(
        WrenProjectConfig(project_path=Path("/tmp/wren-project")),
        toolkit=FakeWrenToolkit([]),
    )

    with pytest.raises(WrenContextError, match="Wren tool wren_query 不存在"):
        client.create_sql_execution_client()


class FakeWrenToolkit:
    """测试用 WrenToolkit，不访问真实 Wren project。"""

    def __init__(self, tools: list["FakeWrenTool"]) -> None:
        self.tools = tools

    def get_tools(self, **_kwargs):
        return self.tools

    def system_prompt(self, *, tools=None) -> str:
        return f"tools={len(tools or [])}"


class FakeWrenTool:
    """测试用 LangChain tool，只记录调用参数并返回固定结果。"""

    def __init__(self, name: str, result: dict) -> None:
        self.name = name
        self.result = result
        self.calls = []

    def invoke(self, arguments: dict):
        self.calls.append(arguments)
        return self.result
