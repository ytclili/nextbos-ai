import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.agent.options import ChatModelOptions
from app.agent.runtime import GraphStreamEvent, run_graph, stream_graph, stream_graph_resume
from app.conversation.repository import ConversationRepository
from app.conversation.summary import extract_running_summary
from app.core.config import Settings

logger = logging.getLogger(__name__)


class AgentService:
    """chat 接口到 LangGraph runtime 之间的应用服务层。"""

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings,
        memory_store=None,
    ):
        self.checkpointer = checkpointer
        self.session_factory = session_factory
        self.settings = settings
        self.memory_store = memory_store
        self.conversation_repository = ConversationRepository(session_factory)

    async def chat(
        self,
        *,
        thread_id: str,
        user_id: str,
        message: str,
        token: str | None = None,
        model_options: ChatModelOptions | None = None,
        trace_id: str | None = None,
    ) -> str:
        """执行一次 chat。

        trace_id 会写入 conversation_messages。
        这样后续可以从聊天记录反查 SigNoZ / OpenTelemetry 调用链路。
        """

        model_options = model_options or ChatModelOptions()
        logger.info(
            "agent.chat.started thread_id=%s user_id=%s input_length=%s",
            thread_id,
            user_id,
            len(message),
        )

        await self._append_user_message(
            thread_id=thread_id,
            user_id=user_id,
            message=message,
            trace_id=trace_id,
        )

        result = await run_graph(
            self.checkpointer,
            thread_id=thread_id,
            user_id=user_id,
            message=message,
            token=token,
            model_options=model_options,
            session_factory=self.session_factory,
            settings=self.settings,
            memory_store=self.memory_store,
        )

        content = _extract_final_content(result)
        await self._persist_assistant_result(
            thread_id=thread_id,
            user_id=user_id,
            content=content,
            trace_id=trace_id,
            result=result,
        )

        logger.info(
            "agent.chat.completed thread_id=%s user_id=%s output_length=%s",
            thread_id,
            user_id,
            len(content),
        )

        return content

    async def stream_chat(
        self,
        *,
        thread_id: str,
        user_id: str,
        message: str,
        token: str | None = None,
        model_options: ChatModelOptions | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """流式执行一次 chat。

        输出给 API 层的是业务事件元组，由 API 层再编码成 SSE。
        数据库仍然只保存完整 user / assistant 消息，不按 token 落库。
        """

        model_options = model_options or ChatModelOptions()
        logger.info(
            "agent.chat.stream.started thread_id=%s user_id=%s input_length=%s",
            thread_id,
            user_id,
            len(message),
        )

        await self._append_user_message(
            thread_id=thread_id,
            user_id=user_id,
            message=message,
            trace_id=trace_id,
        )
        yield (
            "start",
            {
                "code": 200,
                "status": "success",
                "thread_id": thread_id,
                "trace_id": trace_id,
            },
        )

        accumulated_tokens: list[str] = []
        final_state: dict[str, Any] = {}
        interrupted = False

        async for event in stream_graph(
            self.checkpointer,
            thread_id=thread_id,
            user_id=user_id,
            message=message,
            token=token,
            model_options=model_options,
            session_factory=self.session_factory,
            settings=self.settings,
            memory_store=self.memory_store,
        ):
            if event.mode == "messages":
                token = _extract_token(event)
                if token:
                    accumulated_tokens.append(token)
                    yield ("token", {"type": "text", "content": token})
            elif event.mode == "updates":
                for tool_event in _extract_tool_events(event):
                    if _is_interrupt_event(tool_event):
                        interrupted = True
                    yield tool_event
            elif event.mode == "final_state":
                final_state = dict(event.data or {})

        if interrupted:
            content = ""
        else:
            content = _extract_final_content(final_state) or "".join(accumulated_tokens)
            await self._persist_assistant_result(
                thread_id=thread_id,
                user_id=user_id,
                content=content,
                trace_id=trace_id,
                result=final_state,
            )

        logger.info(
            "agent.chat.stream.completed thread_id=%s user_id=%s output_length=%s",
            thread_id,
            user_id,
            len(content),
        )
        yield (
            "done",
            {"content": content, "status": "interrupted"} if interrupted else {"content": content},
        )

    async def stream_resume_chat(
        self,
        *,
        thread_id: str,
        user_id: str,
        resume: dict[str, Any],
        model_options: ChatModelOptions | None = None,
        trace_id: str | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """从 LangGraph interrupt checkpoint 恢复一次流式 chat。"""

        model_options = model_options or ChatModelOptions()
        yield (
            "start",
            {
                "code": 200,
                "status": "success",
                "thread_id": thread_id,
                "trace_id": trace_id,
            },
        )

        accumulated_tokens: list[str] = []
        final_state: dict[str, Any] = {}
        interrupted = False

        async for event in stream_graph_resume(
            self.checkpointer,
            thread_id=thread_id,
            user_id=user_id,
            resume=resume,
            model_options=model_options,
            session_factory=self.session_factory,
            settings=self.settings,
            memory_store=self.memory_store,
        ):
            if event.mode == "messages":
                token = _extract_token(event)
                if token:
                    accumulated_tokens.append(token)
                    yield ("token", {"type": "text", "content": token})
            elif event.mode == "updates":
                for tool_event in _extract_tool_events(event):
                    if _is_interrupt_event(tool_event):
                        interrupted = True
                    yield tool_event
            elif event.mode == "final_state":
                final_state = dict(event.data or {})

        if interrupted:
            content = ""
        else:
            content = _extract_final_content(final_state) or "".join(accumulated_tokens)
            await self._persist_assistant_result(
                thread_id=thread_id,
                user_id=user_id,
                content=content,
                trace_id=trace_id,
                result=final_state,
            )

        yield (
            "done",
            {"content": content, "status": "interrupted"} if interrupted else {"content": content},
        )

    async def _append_user_message(
        self,
        *,
        thread_id: str,
        user_id: str,
        message: str,
        trace_id: str | None,
    ) -> None:
        """保存用户消息。

        这是一个短事务，提交后马上释放数据库连接，不会跨 LLM 调用持有事务。
        """

        await self.conversation_repository.append_user_message(
            thread_id=thread_id,
            user_id=user_id,
            content=message,
            trace_id=trace_id,
        )

    async def _persist_assistant_result(
        self,
        *,
        thread_id: str,
        user_id: str,
        content: str,
        trace_id: str | None,
        result: dict[str, Any],
    ) -> None:
        """保存 assistant 完整回复和可选 summary。"""

        await self.conversation_repository.append_assistant_message(
            thread_id=thread_id,
            user_id=user_id,
            content=content,
            trace_id=trace_id,
            llm_snapshot_id=result.get("llm_snapshot_id"),
        )

        if summary := extract_running_summary(result):
            await self.conversation_repository.save_summary(
                thread_id=thread_id,
                user_id=user_id,
                summary=summary.summary,
                covered_through_message_id=summary.covered_through_message_id,
                message_count=summary.message_count,
            )


def _extract_token(event: GraphStreamEvent) -> str:
    """从 LangGraph messages 事件中提取文本 token。"""

    if not _is_frontend_token_event(event):
        return ""

    chunk = event.data[0] if isinstance(event.data, tuple) and event.data else event.data
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_part_to_text(part) for part in content)
    return str(content) if content else ""


def _extract_tool_events(event: GraphStreamEvent) -> list[tuple[str, dict[str, Any]]]:
    """从 LangGraph updates 事件中提取工具状态事件。"""

    if not isinstance(event.data, dict):
        return []

    events: list[tuple[str, dict[str, Any]]] = []
    events.extend(_interrupt_events(event.data))
    for node_name, update in event.data.items():
        if node_name == "respond":
            events.extend(_tool_start_events(update))
        elif node_name == "tools":
            events.extend(_tool_result_events(update))
    return events


def _interrupt_events(update: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """从 LangGraph __interrupt__ update 中提取前端中断事件。"""

    interrupts = update.get("__interrupt__") or []
    if not isinstance(interrupts, (list, tuple)):
        interrupts = [interrupts]

    events: list[tuple[str, dict[str, Any]]] = []
    for interrupt in interrupts:
        value = getattr(interrupt, "value", None)
        if not isinstance(value, dict):
            continue

        event_type = str(value.get("type") or "interrupt")
        data = dict(value)
        if interrupt_id := getattr(interrupt, "id", None):
            data["interrupt_id"] = str(interrupt_id)
        events.append((event_type, data))
    return events


def _tool_start_events(update: Any) -> list[tuple[str, dict[str, Any]]]:
    """从 respond 节点 update 里提取 tool_start 事件。"""

    events: list[tuple[str, dict[str, Any]]] = []
    for message in _update_messages(update):
        if not isinstance(message, AIMessage):
            continue
        for tool_call in message.tool_calls or []:
            name = str(tool_call.get("name", ""))
            tool_call_id = str(tool_call.get("id", ""))
            if not name:
                continue
            events.append(
                (
                    "tool_start",
                    {
                        "name": name,
                        "tool_call_id": tool_call_id,
                        "message": f"正在调用 {name}",
                    },
                )
            )
    return events


def _tool_result_events(update: Any) -> list[tuple[str, dict[str, Any]]]:
    """从 tools 节点 update 里提取 tool_end / tool_error 事件。"""

    events: list[tuple[str, dict[str, Any]]] = []
    for message in _update_messages(update):
        if not isinstance(message, ToolMessage):
            continue

        name = str(message.name or "")
        tool_call_id = str(message.tool_call_id or "")
        status = str(getattr(message, "status", "success") or "success")
        if status == "error":
            message_text = _message_content_to_text(message.content)
            events.append(
                (
                    "tool_error",
                    {
                        "name": name,
                        "tool_call_id": tool_call_id,
                        "status": status,
                        "message": message_text,
                    },
                )
            )
        else:
            events.append(
                (
                    "tool_end",
                    {
                        "name": name,
                        "tool_call_id": tool_call_id,
                        "status": status,
                    },
                )
            )
    return events


def _update_messages(update: Any) -> list[Any]:
    """从 LangGraph node update 中读取 messages 列表。"""

    if not isinstance(update, dict):
        return []

    messages = update.get("messages") or []
    if isinstance(messages, list):
        return messages
    return [messages]


def _is_interrupt_event(event: tuple[str, dict[str, Any]]) -> bool:
    """判断前端事件是否代表 LangGraph 暂停。"""

    return event[0] == "auth_required" and event[1].get("status") == "interrupted"


def _is_frontend_token_event(event: GraphStreamEvent) -> bool:
    """判断这个 messages 事件是否应该推给前端。

    LangGraph 的 messages stream 会包含图里所有 LLM 节点的 token。
    summarize 是内部压缩上下文用的节点，不能把摘要生成过程展示给用户。
    """

    if not isinstance(event.data, tuple) or len(event.data) < 2:
        return True

    metadata = event.data[1]
    if not isinstance(metadata, dict):
        return True

    node_name = metadata.get("langgraph_node")
    return node_name in {None, "respond", "final_respond"}


def _content_part_to_text(part: Any) -> str:
    """兼容部分供应商返回的结构化 content part。"""

    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        text = part.get("text")
        return text if isinstance(text, str) else ""
    return ""


def _message_content_to_text(content: Any) -> str:
    """把 LangChain message content 转成可展示文本。"""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_part_to_text(part) for part in content)
    return str(content) if content else ""


def _extract_final_content(state: dict[str, Any]) -> str:
    """从最终 LangGraph state 中提取 assistant 完整文本。"""

    messages = state.get("messages") or []
    if not messages:
        return ""

    last_message = messages[-1]
    if isinstance(last_message, BaseMessage):
        content = last_message.content
    else:
        content = getattr(last_message, "content", "")

    return _message_content_to_text(content)
