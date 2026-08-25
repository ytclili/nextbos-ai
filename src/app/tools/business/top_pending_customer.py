import logging
from typing import Literal
from urllib.parse import urlencode

import httpx
from langchain_core.tools import tool

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@tool("get_top_pending_customer")
async def get_top_pending_customer(
    period: Literal["all", "year", "month", "yesterday", "today"] = "yesterday",
) -> dict | None:
    """查询本商户指定期间待收款最高的客户。"""

    settings = get_settings()
    query = urlencode({"period": period})
    path = f"/api/ai-tools/customers/top-pending?{query}"
    url = f"{settings.core_internal_base_url.rstrip('/')}{path}"

    # 第一版只使用 mock token 透传给核心业务系统。
    # 后续如果核心系统改成 HMAC 或其它内部鉴权，只需要替换这里的 headers。
    headers = {}
    if settings.core_internal_token:
        token = settings.core_internal_token.strip()
        headers["Authorization"] = (
            token if token.lower().startswith("bearer ") else f"Bearer {token}"
        )

    logger.info(
        (
            "business.tool.get_top_pending_customer.request url=%s period=%s "
            "token_set=%s"
        ),
        url,
        period,
        bool(headers.get("Authorization")),
    )

    async with httpx.AsyncClient(timeout=settings.core_internal_timeout_seconds) as client:
        response = await client.get(url, headers=headers)

    logger.info(
        "business.tool.get_top_pending_customer.response status_code=%s body=%s",
        response.status_code,
        _short_text(response.text),
    )

    response.raise_for_status()

    if not response.content:
        logger.info("business.tool.get_top_pending_customer.result data=None")
        return None

    envelope = response.json()
    result = envelope.get("data", envelope) if isinstance(envelope, dict) else envelope
    logger.info(
        "business.tool.get_top_pending_customer.result data=%s",
        _short_text(str(result)),
    )
    return result


def _short_text(value: str, *, max_length: int = 2000) -> str:
    """把响应内容截断后写日志，避免控制台被大响应刷屏。"""

    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}...(truncated, length={len(value)})"
