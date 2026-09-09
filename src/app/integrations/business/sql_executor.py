import asyncio
from typing import Any, Protocol

from app.agent.schemas.sql_execution import (
    SqlExecutionColumn,
    SqlExecutionError,
    SqlExecutionResult,
)


class SqlExecutionClient(Protocol):
    """受控 SQL 执行客户端协议。"""

    async def execute_sql(
        self,
        *,
        sql: str,
        query_id: str | None = None,
        max_rows: int = 100,
    ) -> SqlExecutionResult:
        """执行已经通过校验的 SQL，并返回标准数据集快照。"""


class LangChainToolSqlExecutionClient:
    """基于 LangChain tool 的 SQL 执行客户端。

    第一阶段只适配类似 Wren `wren_query` 的工具输出；
    真正权限策略、参数化执行、超时和审计在后续节点补齐。
    """

    def __init__(self, query_tool: Any) -> None:
        self.query_tool = query_tool

    async def execute_sql(
        self,
        *,
        sql: str,
        query_id: str | None = None,
        max_rows: int = 100,
    ) -> SqlExecutionResult:
        """在线程池中调用同步 LangChain tool，避免阻塞 async LangGraph。"""

        return await asyncio.to_thread(
            self.execute_sql_sync,
            sql=sql,
            query_id=query_id,
            max_rows=max_rows,
        )

    def execute_sql_sync(
        self,
        *,
        sql: str,
        query_id: str | None = None,
        max_rows: int = 100,
    ) -> SqlExecutionResult:
        """调用底层 query tool，并转换成 SqlExecutionResult。"""

        try:
            payload = self.query_tool.invoke({"sql": sql, "limit": max_rows})
        except Exception as exc:
            return _failed_execution_result(sql=sql, query_id=query_id, message=str(exc))

        if not isinstance(payload, dict):
            return _failed_execution_result(
                sql=sql,
                query_id=query_id,
                message="SQL 执行工具返回内容不是对象",
            )

        if payload.get("ok") is False:
            return _failed_execution_result(
                sql=sql,
                query_id=query_id,
                message=_error_message(payload),
            )

        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        rows = _rows_from_payload(data)
        columns = _columns_from_payload(data, rows)
        truncated = bool(data.get("truncated", False)) if isinstance(data, dict) else False

        return SqlExecutionResult(
            status="succeeded",
            query_id=query_id,
            sql=sql,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
        )


def _rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """从执行工具 payload 中读取结果行。"""

    raw_rows = payload.get("rows") or payload.get("results") or []
    if not isinstance(raw_rows, list):
        return []
    return [row for row in raw_rows if isinstance(row, dict)]


def _columns_from_payload(
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[SqlExecutionColumn]:
    """优先读取工具返回的 columns；缺失时从第一行推断字段名。"""

    raw_columns = payload.get("columns") or []
    if isinstance(raw_columns, list) and raw_columns:
        return [_column_from_payload(column) for column in raw_columns]

    if not rows:
        return []

    return [SqlExecutionColumn(name=name) for name in rows[0]]


def _column_from_payload(column: Any) -> SqlExecutionColumn:
    """兼容字符串列名和对象列描述。"""

    if isinstance(column, str):
        return SqlExecutionColumn(name=column)
    if isinstance(column, dict):
        name = str(column.get("name") or column.get("column_name") or "")
        data_type = column.get("data_type") or column.get("type")
        return SqlExecutionColumn(name=name, data_type=str(data_type) if data_type else None)
    return SqlExecutionColumn(name=str(column))


def _failed_execution_result(
    *,
    sql: str,
    query_id: str | None,
    message: str,
) -> SqlExecutionResult:
    """构造失败执行结果，避免后续节点误读半截数据。"""

    return SqlExecutionResult(
        status="failed",
        query_id=query_id,
        sql=sql,
        error=SqlExecutionError(error_type="database", message=message),
    )


def _error_message(payload: dict[str, Any]) -> str:
    """从执行工具 envelope 中提取紧凑错误。"""

    error = payload.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    if error:
        return str(error)
    if payload.get("content"):
        return str(payload["content"])
    return "SQL 执行失败"
