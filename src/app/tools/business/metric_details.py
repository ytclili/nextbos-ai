import logging
from typing import Literal
from urllib.parse import urlencode

import httpx
from langchain_core.tools import tool

from app.core.config import get_settings
from app.tools.business.auth import core_internal_headers

logger = logging.getLogger(__name__)


@tool("get_metric_details")
async def get_metric_details(
    metric: Literal["receivable", "received", "pending"],
    period: Literal["all", "year", "month", "yesterday", "today"] = "yesterday",
    page: int = 1,
    page_size: int = 50,
    sort_by: str | None = None,
) -> dict:
    """查询应收、已收、待收指标明细。"""

    settings = get_settings()
    query = [
        ("metric", metric),
        ("period", period),
        ("page", str(page)),
        ("pageSize", str(page_size)),
    ]
    if sort_by is not None:
        query.append(("sortBy", sort_by))

    path = f"/api/ai-tools/metric-details?{urlencode(query)}"
    url = f"{settings.core_internal_base_url.rstrip('/')}{path}"
    headers = core_internal_headers(settings)

    logger.info(
        (
            "business.tool.get_metric_details.request url=%s metric=%s period=%s "
            "page=%s page_size=%s token_set=%s"
        ),
        url,
        metric,
        period,
        page,
        page_size,
        bool(headers.get("Authorization")),
    )

    async with httpx.AsyncClient(timeout=settings.core_internal_timeout_seconds) as client:
        response = await client.get(url, headers=headers)

    logger.info(
        "business.tool.get_metric_details.response status_code=%s body=%s",
        response.status_code,
        _short_text(response.text),
    )

    response.raise_for_status()

    if not response.content:
        logger.info("business.tool.get_metric_details.result data={}")
        return {}

    envelope = response.json()
    result = envelope.get("data", envelope) if isinstance(envelope, dict) else envelope
    logger.info("business.tool.get_metric_details.result data=%s", _short_text(str(result)))
    return result


def _short_text(value: str, *, max_length: int = 2000) -> str:
    """把响应内容截断后写日志，避免控制台被大响应刷屏。"""

    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}...(truncated, length={len(value)})"
