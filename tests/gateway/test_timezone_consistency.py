"""Tests for timezone-aware datetime consistency."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from any_llm.gateway.api.routes.chat import log_usage
from any_llm.gateway.models.entities import UsageLog
from any_llm.gateway.services.log_writer import LogWriter


@pytest.mark.asyncio
async def test_usage_log_timestamp_is_timezone_aware(test_db: AsyncSession, log_writer: LogWriter) -> None:
    """Test that usage log timestamps are stored with timezone info."""
    await log_usage(
        db=test_db,
        log_writer=log_writer,
        api_key_obj=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        error="test error",
    )

    log = (await test_db.execute(select(UsageLog))).scalars().first()
    assert log is not None
    assert log.timestamp is not None
    # The timestamp should be timezone-aware (has tzinfo)
    assert log.timestamp.tzinfo is not None, "Timestamp should be timezone-aware"
