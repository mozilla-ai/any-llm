"""Tests for timezone-aware datetime consistency."""

import pytest
from sqlalchemy.orm import Session

from any_llm.gateway.db.models import UsageLog
from any_llm.gateway.routes.chat import _log_usage


@pytest.mark.asyncio
async def test_usage_log_timestamp_is_timezone_aware(test_db: Session) -> None:
    """Test that usage log timestamps are stored with timezone info."""
    await _log_usage(
        db=test_db,
        api_key_obj=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        error="test error",
    )

    log = test_db.query(UsageLog).first()
    assert log is not None
    assert log.timestamp is not None
    # The timestamp should be timezone-aware (has tzinfo)
    assert log.timestamp.tzinfo is not None, "Timestamp should be timezone-aware"
