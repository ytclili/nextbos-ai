from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage
from pydantic import ValidationError

from app.agent.schemas.sql_execution import SqlExecutionError, SqlExecutionResult
from app.agent.schemas.sql_validation import SqlValidationResult
from app.agent.state import AgentState
from app.integrations.business.sql_executor import SqlExecutionClient

SqlExecuteNode = Callable[[AgentState], Awaitable[dict[str, Any]]]


def create_sql_execute_node(
    *,
    sql_execution_client: SqlExecutionClient | None = None,
    max_rows: int = 100,
) -> SqlExecuteNode:
    """创建 SQL 执行节点。

    这个节点只执行已经通过 Wren dry-plan / dry-run 的 validated_sql；
    未通过校验、缺少执行客户端或校验快照异常时都会跳过执行。
    """

    async def sql_execute(state: AgentState) -> dict[str, Any]:
        """执行 validated_sql，并把 SqlExecutionResult 写入 LangGraph state。"""

        validation_result = _sql_validation_result_from_state(state)

        if validation_result is None:
            result = _skipped_execution_result("SQL 校验结果为空或格式不合法")
        elif validation_result.status != "passed" or not validation_result.validated_sql:
            result = _skipped_execution_result("SQL 未通过校验，已跳过执行")
        elif sql_execution_client is None:
            result = _skipped_execution_result("未配置 SQL 执行客户端")
        else:
            result = await sql_execution_client.execute_sql(
                sql=validation_result.validated_sql,
                query_id=str(uuid4()),
                max_rows=max_rows,
            )

        return {
            "sql_execution_result": result.model_dump(),
            "sql_result_review": _review_sql_execution_result(result),
            "messages": [AIMessage(content=_execution_message(result))],
        }

    return sql_execute


def _sql_validation_result_from_state(state: AgentState) -> SqlValidationResult | None:
    """从 state 读取 SQL 校验快照；格式异常时返回 None，避免图直接崩溃。"""

    try:
        return SqlValidationResult.model_validate(state.get("sql_validation_result") or {})
    except ValidationError:
        return None


def _skipped_execution_result(message: str) -> SqlExecutionResult:
    """构造跳过执行的结果快照。"""

    return SqlExecutionResult(
        status="skipped",
        error=SqlExecutionError(error_type="validation", message=message),
    )


def _execution_message(result: SqlExecutionResult) -> str:
    """把 SQL 执行状态转换成 Studio 可读消息。"""

    if result.status == "succeeded":
        return f"SQL 执行成功，返回 {result.row_count} 行。"
    if result.status == "skipped":
        return f"SQL 执行已跳过：{result.error.message if result.error else '未满足执行条件'}"
    return f"SQL 执行失败：{result.error.message if result.error else '未知错误'}"


def _review_sql_execution_result(result: SqlExecutionResult) -> dict[str, Any]:
    """标记需要复核的执行结果，供后续 SQL 修复路由使用。"""

    needs_review = result.status == "succeeded" and _has_empty_or_null_result(result)
    return {
        "needs_review": needs_review,
        "reason": "SQL 执行成功但结果为空或全为空值" if needs_review else None,
    }


def _has_empty_or_null_result(result: SqlExecutionResult) -> bool:
    """判断结果是否没有可用于回答的有效值。"""

    if result.row_count == 0 or not result.rows:
        return True
    return all(value is None for row in result.rows for value in row.values())
