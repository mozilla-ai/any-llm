"""Bulk usage log endpoint.

Provides a single query interface over all usage logs with optional
time range and user filters, ordered newest-first. Intended for
external systems that need to sync usage data (billing, analytics).
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from any_llm.gateway.api.deps import get_db, verify_master_key
from any_llm.gateway.models.entities import UsageLog
from any_llm.utils.timestamp import to_datetime

router = APIRouter(prefix="/v1/usage", tags=["usage"])


class UsageEntry(BaseModel):
    """A single usage log entry."""

    id: str
    user_id: str | None
    api_key_id: str | None
    timestamp: str
    model: str
    provider: str | None
    endpoint: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost: float | None
    status: str
    error_message: str | None

    @classmethod
    def from_model(cls, log: UsageLog) -> "UsageEntry":
        return cls(
            id=log.id,
            user_id=log.user_id,
            api_key_id=log.api_key_id,
            timestamp=log.timestamp.isoformat(),
            model=log.model,
            provider=log.provider,
            endpoint=log.endpoint,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            cost=log.cost,
            status=log.status,
            error_message=log.error_message,
        )


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_usage(
    db: Annotated[Session, Depends(get_db)],
    start_date: int | datetime | None = Query(
        default=None,
        description="Return logs with timestamp >= start_date (ISO 8601 or Unix epoch seconds)",
    ),
    end_date: int | datetime | None = Query(
        default=None,
        description="Return logs with timestamp < end_date (ISO 8601 or Unix epoch seconds)",
    ),
    user_id: str | None = Query(default=None, description="Filter to a single user"),
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[UsageEntry]:
    """List usage logs ordered by timestamp (most recent first).

    Supports optional filters for time range and user. Paginated via skip/limit.
    Timestamps accept either ISO 8601 strings or Unix epoch seconds (int).
    """
    start = to_datetime(start_date)
    end = to_datetime(end_date)

    query = db.query(UsageLog)
    if start is not None:
        query = query.filter(UsageLog.timestamp >= start)
    if end is not None:
        query = query.filter(UsageLog.timestamp < end)
    if user_id is not None:
        query = query.filter(UsageLog.user_id == user_id)

    logs = query.order_by(UsageLog.timestamp.desc()).offset(skip).limit(limit).all()
    return [UsageEntry.from_model(log) for log in logs]
