"""Tests for usage logging commit scope isolation."""

import pytest
from sqlalchemy.orm import Session

from any_llm.gateway.db.models import UsageLog
from any_llm.gateway.routes.chat import _log_usage
from any_llm.types.completion import CompletionUsage


@pytest.mark.asyncio
async def test_log_usage_creates_usage_log(test_db: Session) -> None:
    """Test that _log_usage successfully creates a usage log entry."""
    usage = CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    await _log_usage(
        db=test_db,
        api_key_obj=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        usage_override=usage,
    )

    log = test_db.query(UsageLog).first()
    assert log is not None
    assert log.prompt_tokens == 100
    assert log.completion_tokens == 50
    assert log.status == "success"


@pytest.mark.asyncio
async def test_log_usage_records_error(test_db: Session) -> None:
    """Test that _log_usage records error status and message."""
    await _log_usage(
        db=test_db,
        api_key_obj=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        error="Provider timeout",
    )

    log = test_db.query(UsageLog).first()
    assert log is not None
    assert log.status == "error"
    assert log.error_message == "Provider timeout"
