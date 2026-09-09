import pytest

from app.integrations.business.sql_executor import LangChainToolSqlExecutionClient


class FakeQueryTool:
    """测试用 LangChain query tool。"""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def invoke(self, arguments):
        self.calls.append(arguments)
        return self.payload


class FailingQueryTool:
    """测试用异常 query tool。"""

    def invoke(self, arguments):
        raise RuntimeError("数据库连接失败")


@pytest.mark.asyncio
async def test_langchain_tool_sql_execution_client_returns_dataset() -> None:
    """执行工具成功时，应该返回标准数据集。"""

    tool = FakeQueryTool(
        {
            "ok": True,
            "data": {
                "columns": [
                    {"name": "goods_id", "data_type": "BIGINT"},
                    {"name": "goods_name", "data_type": "VARCHAR"},
                ],
                "rows": [
                    {"goods_id": 1, "goods_name": "苹果"},
                    {"goods_id": 2, "goods_name": "香蕉"},
                ],
                "truncated": False,
            },
        }
    )
    client = LangChainToolSqlExecutionClient(tool)

    result = await client.execute_sql(
        sql="select id as goods_id, description as goods_name from wms_items",
        query_id="query-1",
        max_rows=20,
    )

    assert tool.calls == [
        {
            "sql": "select id as goods_id, description as goods_name from wms_items",
            "limit": 20,
        }
    ]
    assert result.status == "succeeded"
    assert result.query_id == "query-1"
    assert [column.name for column in result.columns] == ["goods_id", "goods_name"]
    assert result.rows[0]["goods_name"] == "苹果"
    assert result.row_count == 2
    assert result.truncated is False


def test_langchain_tool_sql_execution_client_infers_columns_from_rows() -> None:
    """工具没有返回 columns 时，应该从第一行推断字段名。"""

    tool = FakeQueryTool(
        {
            "rows": [
                {"goods_id": 1, "goods_name": "苹果"},
            ],
        }
    )
    client = LangChainToolSqlExecutionClient(tool)

    result = client.execute_sql_sync(sql="select 1")

    assert result.status == "succeeded"
    assert [column.name for column in result.columns] == ["goods_id", "goods_name"]
    assert result.row_count == 1


def test_langchain_tool_sql_execution_client_accepts_results_key() -> None:
    """部分工具用 results 表示数据行，也应该兼容。"""

    tool = FakeQueryTool(
        {
            "ok": True,
            "data": {
                "columns": ["order_id", "amount"],
                "results": [
                    {"order_id": 1, "amount": "100.00"},
                ],
            },
        }
    )
    client = LangChainToolSqlExecutionClient(tool)

    result = client.execute_sql_sync(sql="select 1")

    assert result.status == "succeeded"
    assert [column.name for column in result.columns] == ["order_id", "amount"]
    assert result.rows == [{"order_id": 1, "amount": "100.00"}]


def test_langchain_tool_sql_execution_client_returns_failed_when_tool_fails() -> None:
    """执行工具抛异常时，应该返回失败结果，不让 graph 崩溃。"""

    client = LangChainToolSqlExecutionClient(FailingQueryTool())

    result = client.execute_sql_sync(sql="select 1")

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.message == "数据库连接失败"


def test_langchain_tool_sql_execution_client_returns_failed_for_error_envelope() -> None:
    """工具返回 ok=false 时，应该转换成失败结果。"""

    tool = FakeQueryTool(
        {
            "ok": False,
            "error": {"message": "权限不足"},
        }
    )
    client = LangChainToolSqlExecutionClient(tool)

    result = client.execute_sql_sync(sql="select 1")

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.message == "权限不足"


def test_langchain_tool_sql_execution_client_returns_failed_for_invalid_payload() -> None:
    """工具返回非对象时，应该转换成失败结果。"""

    client = LangChainToolSqlExecutionClient(FakeQueryTool("bad-payload"))

    result = client.execute_sql_sync(sql="select 1")

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.message == "SQL 执行工具返回内容不是对象"
