"""Tests for usage logging commit scope isolation."""

from unittest.mock import patch

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from any_llm.gateway.api.routes.chat import log_usage
from any_llm.gateway.models.entities import UsageLog
from any_llm.types.completion import CompletionUsage


@pytest.mark.asyncio
async def test_log_usage_creates_usage_log(test_db: Session) -> None:
    """Test that log_usage successfully creates a usage log entry."""
    usage = CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

    await log_usage(
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
    """Test that log_usage records error status and message."""
    await log_usage(
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


@pytest.mark.asyncio
async def test_log_usage_does_not_use_savepoint(test_db: Session) -> None:
    """Test that log_usage commits directly without a savepoint."""
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    with patch.object(test_db, "begin_nested", wraps=test_db.begin_nested) as mock_nested:
        await log_usage(
            db=test_db,
            api_key_obj=None,
            model="gpt-4o",
            provider="openai",
            endpoint="/v1/chat/completions",
            usage_override=usage,
        )
        mock_nested.assert_not_called()

    log = test_db.query(UsageLog).first()
    assert log is not None
    assert log.total_tokens == 15


@pytest.mark.asyncio
async def test_log_usage_rollback_on_commit_failure(test_db: Session) -> None:
    """Test that log_usage rolls back cleanly when commit fails."""
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    with (
        patch.object(test_db, "commit", side_effect=OperationalError("db", {}, Exception("db gone"))),
        patch.object(test_db, "rollback", wraps=test_db.rollback) as mock_rollback,
    ):
        await log_usage(
            db=test_db,
            api_key_obj=None,
            model="gpt-4o",
            provider="openai",
            endpoint="/v1/chat/completions",
            usage_override=usage,
        )
        mock_rollback.assert_called_once()

    log = test_db.query(UsageLog).first()
    assert log is None
