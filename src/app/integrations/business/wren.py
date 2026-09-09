import asyncio
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.agent.schemas.sql_validation import SqlValidationIssue, SqlValidationResult
from app.integrations.business.sql_executor import (
    LangChainToolSqlExecutionClient,
    SqlExecutionClient,
)


class WrenContextFilter(BaseModel):
    """传给 WrenAI 的单个业务过滤条件。"""

    field: str = Field(description="业务字段或概念，例如区域、渠道、商品。")
    value: str = Field(description="过滤值，例如华东、小程序、某个商品名。")


class WrenContextRequest(BaseModel):
    """WrenAI 上下文查询请求。

    这里承接 intent 节点已经识别出的业务问题结构，不在这一层生成 SQL。
    """

    question: str = Field(description="当前需要分析或查询的用户问题。")
    tenant_id: str | None = Field(default=None, description="租户标识，用于后续隔离业务模型。")
    user_id: str | None = Field(default=None, description="用户标识，用于审计和个性化上下文。")
    metrics: list[str] = Field(default_factory=list, description="用户关心的指标。")
    dimensions: list[str] = Field(default_factory=list, description="用户要求分组或下钻的维度。")
    filters: list[WrenContextFilter] = Field(
        default_factory=list,
        description="用户明确给出的过滤条件。",
    )
    time_range: str | None = Field(default=None, description="用户明确或可推断的时间范围。")


class WrenProjectConfig(BaseModel):
    """官方 wren-langchain toolkit 的项目配置。"""

    project_path: Path = Field(description="Wren project 根目录，目录下需要有 wren_project.yml。")
    profile: str | None = Field(default=None, description="可选 Wren profile 名称。")
    context_limit: int = Field(default=5, ge=1, le=20, description="上下文召回数量。")
    recall_limit: int = Field(default=3, ge=0, le=10, description="历史 SQL 召回数量。")
    include_memory_write: bool = Field(
        default=False,
        description="是否暴露 wren_store_query。业务查询链路默认只读，避免调试期污染 memory。",
    )
    raise_on_error: bool = Field(default=True, description="Wren 工具调用失败时是否抛出异常。")


class WrenModel(BaseModel):
    """WrenAI 返回的业务模型信息。"""

    name: str
    description: str | None = None


class WrenField(BaseModel):
    """WrenAI 返回的字段信息。"""

    model: str
    name: str
    data_type: str | None = None
    description: str | None = None


class WrenMetric(BaseModel):
    """WrenAI 返回的指标定义。"""

    name: str
    expression: str | None = None
    description: str | None = None


class WrenJoin(BaseModel):
    """WrenAI 返回的模型 Join 关系。"""

    left_model: str
    right_model: str
    condition: str
    description: str | None = None


class WrenBusinessRule(BaseModel):
    """WrenAI 返回的业务规则。"""

    name: str
    description: str


class WrenSqlExample(BaseModel):
    """WrenAI 返回的历史问题和 SQL 示例。"""

    question: str
    sql: str
    note: str | None = None


class WrenContextResult(BaseModel):
    """后续 SQL 规划节点可使用的 WrenAI 上下文快照。"""

    models: list[WrenModel] = Field(default_factory=list)
    fields: list[WrenField] = Field(default_factory=list)
    metrics: list[WrenMetric] = Field(default_factory=list)
    joins: list[WrenJoin] = Field(default_factory=list)
    business_rules: list[WrenBusinessRule] = Field(default_factory=list)
    historical_sql: list[WrenSqlExample] = Field(default_factory=list)
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description="保留原始响应，便于调试；业务节点优先使用上面的标准化字段。",
    )


class WrenContextClient(Protocol):
    """WrenAI 上下文查询客户端协议。"""

    async def query_context(self, request: WrenContextRequest) -> WrenContextResult:
        """根据业务问题查询 WrenAI 上下文。"""

    async def validate_sql(
        self,
        *,
        candidate_name: str,
        sql: str,
        repair_attempt: int = 0,
        max_repair_attempts: int = 2,
    ) -> SqlValidationResult:
        """通过 Wren dry-plan / dry-run 校验候选 SQL。"""


class WrenContextError(RuntimeError):
    """WrenAI 上下文查询失败。"""


class WrenLangChainNotInstalledError(WrenContextError):
    """当前环境没有安装官方 wren-langchain。"""


class WrenLangChainContextClient:
    """基于官方 wren-langchain WrenToolkit 的上下文客户端。"""

    def __init__(
        self,
        config: WrenProjectConfig,
        *,
        toolkit: Any | None = None,
    ) -> None:
        self.config = config
        self._toolkit = toolkit
        self._tools: list[Any] | None = None

    def get_tools(self) -> list[Any]:
        """返回官方 Wren LangChain tools，供后续 agent 节点绑定。"""

        if self._tools is None:
            self._tools = self.toolkit.get_tools(
                include_memory_write=self.config.include_memory_write,
                raise_on_error=self.config.raise_on_error,
            )
        return self._tools

    def system_prompt(self) -> str:
        """返回和当前 Wren tools 同步的官方系统提示词。"""

        tools = self.get_tools()
        return self.toolkit.system_prompt(tools=tools)

    def create_sql_execution_client(self) -> SqlExecutionClient:
        """基于官方 wren_query tool 创建 SQL 执行客户端。"""

        query_tool = _required_tool(self.get_tools(), "wren_query")
        return LangChainToolSqlExecutionClient(query_tool)

    async def query_context(self, request: WrenContextRequest) -> WrenContextResult:
        """在线程池中调用官方同步 SDK，避免阻塞 async LangGraph。"""

        return await asyncio.to_thread(self.query_context_sync, request)

    async def validate_sql(
        self,
        *,
        candidate_name: str,
        sql: str,
        repair_attempt: int = 0,
        max_repair_attempts: int = 2,
    ) -> SqlValidationResult:
        """在线程池中调用官方 dry-plan，避免阻塞 async LangGraph。"""

        return await asyncio.to_thread(
            self.validate_sql_sync,
            candidate_name=candidate_name,
            sql=sql,
            repair_attempt=repair_attempt,
            max_repair_attempts=max_repair_attempts,
        )

    def query_context_sync(self, request: WrenContextRequest) -> WrenContextResult:
        """通过官方 LangChain tools 查询 Wren 上下文。

        轻量安装时通常只有 wren_query / wren_dry_plan / wren_list_models；
        如果项目已启用 .wren/memory/，官方 toolkit 会自动额外暴露
        wren_fetch_context / wren_recall_queries。
        """

        try:
            tools = self.get_tools()
            list_models_payload = _invoke_optional_tool(tools, "wren_list_models", {})
            context_payload = _invoke_optional_tool(
                tools,
                "wren_fetch_context",
                {
                    "question": request.question,
                    "limit": self.config.context_limit,
                },
            )
            recall_payload = (
                {}
                if self.config.recall_limit == 0
                else _invoke_optional_tool(
                    tools,
                    "wren_recall_queries",
                    {
                        "question": request.question,
                        "limit": self.config.recall_limit,
                    },
                )
            )
        except Exception as exc:
            raise WrenContextError("WrenAI 官方 SDK 查询上下文失败") from exc

        return _context_result_from_wren_payload(
            context_payload,
            list_models_payload=list_models_payload,
            recalled_queries=_tool_data_results(recall_payload),
        )

    def validate_sql_sync(
        self,
        *,
        candidate_name: str,
        sql: str,
        repair_attempt: int = 0,
        max_repair_attempts: int = 2,
    ) -> SqlValidationResult:
        """通过官方 wren_dry_plan 校验候选 SQL，返回内部校验快照。"""

        try:
            payload = _invoke_required_tool_envelope(
                self.get_tools(),
                "wren_dry_plan",
                {"sql": sql},
            )
        except Exception as exc:
            return _failed_sql_validation_result(
                candidate_name=candidate_name,
                sql=sql,
                message=str(exc),
                repair_attempt=repair_attempt,
                max_repair_attempts=max_repair_attempts,
            )

        if payload.get("ok") is False:
            return _failed_sql_validation_result(
                candidate_name=candidate_name,
                sql=sql,
                message=_wren_error_message(payload),
                dry_plan=payload,
                repair_attempt=repair_attempt,
                max_repair_attempts=max_repair_attempts,
            )

        data = payload.get("data", {})
        dialect_sql = data.get("dialect_sql") if isinstance(data, dict) else None
        if not dialect_sql:
            return _failed_sql_validation_result(
                candidate_name=candidate_name,
                sql=sql,
                message="Wren dry-plan 未返回 dialect_sql",
                dry_plan=payload,
                repair_attempt=repair_attempt,
                max_repair_attempts=max_repair_attempts,
            )

        return SqlValidationResult(
            candidate_name=candidate_name,
            sql=sql,
            status="passed",
            validated_sql=str(dialect_sql),
            dry_plan=payload,
            repair_attempt=repair_attempt,
            max_repair_attempts=max_repair_attempts,
        )

    @property
    def toolkit(self) -> Any:
        """延迟创建官方 WrenToolkit，避免未配置 Wren 项目时影响服务启动。"""

        if self._toolkit is None:
            self._toolkit = _load_wren_toolkit().from_project(
                self.config.project_path,
                profile=self.config.profile,
            )
        return self._toolkit


def _load_wren_toolkit() -> Any:
    """加载官方 wren-langchain WrenToolkit。"""

    try:
        from wren_langchain import WrenToolkit  # noqa: PLC0415
    except ImportError as exc:
        raise WrenLangChainNotInstalledError(
            "未安装官方 wren-langchain，请先安装匹配数据源的 extra。"
        ) from exc
    return WrenToolkit


def _context_result_from_wren_payload(
    context_payload: dict[str, Any],
    *,
    list_models_payload: dict[str, Any],
    recalled_queries: list[dict[str, Any]],
) -> WrenContextResult:
    """把官方 Wren payload 转成 agent 内部标准上下文。"""

    return WrenContextResult(
        models=(
            _models_from_context(context_payload)
            + _models_from_list_models(list_models_payload)
        ),
        fields=_fields_from_context(context_payload),
        joins=_joins_from_context(context_payload),
        historical_sql=_sql_examples_from_recalled_queries(recalled_queries),
        raw={
            "context": context_payload,
            "list_models": list_models_payload,
            "recalled_queries": recalled_queries,
        },
    )


def _invoke_optional_tool(
    tools: list[Any],
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """调用官方 Wren tool；工具不存在时返回空 payload。"""

    tool = _tool_by_name(tools, tool_name)
    if tool is None:
        return {}
    result = tool.invoke(arguments)
    if not isinstance(result, dict):
        raise WrenContextError(f"Wren tool {tool_name} 返回内容不是对象")
    if result.get("ok") is False:
        raise WrenContextError(f"Wren tool {tool_name} 调用失败：{result.get('error')}")
    data = result.get("data")
    return data if isinstance(data, dict) else result


def _invoke_required_tool_envelope(
    tools: list[Any],
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """调用必需的官方 Wren tool，并保留完整 envelope。"""

    tool = _tool_by_name(tools, tool_name)
    if tool is None:
        raise WrenContextError(f"Wren tool {tool_name} 不存在")
    result = tool.invoke(arguments)
    if not isinstance(result, dict):
        raise WrenContextError(f"Wren tool {tool_name} 返回内容不是对象")
    return result


def _tool_by_name(tools: list[Any], tool_name: str) -> Any | None:
    """按 LangChain tool name 查找官方 Wren tool。"""

    for tool in tools:
        if getattr(tool, "name", None) == tool_name:
            return tool
    return None


def _required_tool(tools: list[Any], tool_name: str) -> Any:
    """读取必需的官方 Wren tool；不存在时明确失败。"""

    tool = _tool_by_name(tools, tool_name)
    if tool is None:
        raise WrenContextError(f"Wren tool {tool_name} 不存在")
    return tool


def _tool_data_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """读取 Wren tool envelope data.results。"""

    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    return [row for row in results if isinstance(row, dict)]


def _models_from_context(payload: dict[str, Any]) -> list[WrenModel]:
    """从 Wren context 里抽取模型条目。"""

    models = [
        WrenModel(name=item["name"], description=_context_item_description(item))
        for item in _context_items(payload, "model")
        if item.get("name")
    ]
    return models + [WrenModel.model_validate(item) for item in payload.get("models", [])]


def _models_from_list_models(payload: dict[str, Any]) -> list[WrenModel]:
    """从 wren_list_models 工具结果里抽取模型条目。"""

    models = payload.get("models", [])
    if not isinstance(models, list):
        return []
    return [
        WrenModel(
            name=item["name"],
            description=item.get("description"),
        )
        for item in models
        if isinstance(item, dict) and item.get("name")
    ]


def _fields_from_context(payload: dict[str, Any]) -> list[WrenField]:
    """从 Wren context 里抽取字段条目。"""

    fields = []
    for item in _context_items(payload, "column"):
        model = item.get("model") or item.get("model_name") or item.get("parent_name")
        if model and item.get("name"):
            fields.append(
                WrenField(
                    model=model,
                    name=item["name"],
                    data_type=item.get("data_type"),
                    description=_context_item_description(item),
                )
            )
    return fields + [WrenField.model_validate(item) for item in payload.get("fields", [])]


def _joins_from_context(payload: dict[str, Any]) -> list[WrenJoin]:
    """从 Wren context 里抽取关系条目。"""

    joins = []
    for item in _context_items(payload, "relationship"):
        left_model = item.get("left_model") or item.get("left")
        right_model = item.get("right_model") or item.get("right")
        condition = item.get("condition") or item.get("text") or item.get("summary")
        if left_model and right_model and condition:
            joins.append(
                WrenJoin(
                    left_model=left_model,
                    right_model=right_model,
                    condition=condition,
                    description=_context_item_description(item),
                )
            )
    return joins + [WrenJoin.model_validate(item) for item in payload.get("joins", [])]


def _context_items(payload: dict[str, Any], item_type: str) -> list[dict[str, Any]]:
    """读取 Wren memory fetch 返回的指定类型结果。"""

    items = payload.get("results", [])
    if not isinstance(items, list):
        return []
    return [
        item
        for item in items
        if isinstance(item, dict) and item.get("item_type") == item_type
    ]


def _context_item_description(item: dict[str, Any]) -> str | None:
    """提取 Wren context 条目的描述文本。"""

    value = item.get("summary") or item.get("description") or item.get("text")
    return str(value) if value is not None else None


def _sql_examples_from_recalled_queries(rows: list[dict[str, Any]]) -> list[WrenSqlExample]:
    """把 Wren recall 结果转换成历史 SQL 示例。"""

    examples = []
    for row in rows:
        question = row.get("nl_query") or row.get("nl") or row.get("question")
        sql = row.get("sql_query") or row.get("sql")
        if question and sql:
            examples.append(
                WrenSqlExample(
                    question=str(question),
                    sql=str(sql),
                    note=row.get("note"),
                )
            )
    return examples


def _failed_sql_validation_result(
    *,
    candidate_name: str,
    sql: str,
    message: str,
    dry_plan: dict[str, Any] | None = None,
    repair_attempt: int,
    max_repair_attempts: int,
) -> SqlValidationResult:
    """把 Wren dry-plan 失败转换成后续 SQL 修复节点可读取的快照。"""

    status = "failed" if repair_attempt >= max_repair_attempts else "needs_repair"
    return SqlValidationResult(
        candidate_name=candidate_name,
        sql=sql,
        status=status,
        issues=[
            SqlValidationIssue(
                issue_type="unknown",
                severity="error",
                message=message,
            )
        ],
        dry_plan=dry_plan or {},
        repair_attempt=repair_attempt,
        max_repair_attempts=max_repair_attempts,
    )


def _wren_error_message(payload: dict[str, Any]) -> str:
    """从官方 Wren error envelope 中提取紧凑错误。"""

    error = payload.get("error")
    if isinstance(error, dict) and error.get("message"):
        return str(error["message"])
    content = payload.get("content")
    if content:
        return str(content)
    return "Wren dry-plan 校验失败"
