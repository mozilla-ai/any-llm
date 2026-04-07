"""Tests for budget enforcement strategies."""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from any_llm.gateway.models.entities import Budget, User
from any_llm.gateway.services import budget_service
from any_llm.gateway.services.budget_service import validate_user_budget


@pytest.mark.asyncio
async def test_validate_user_budget_uses_for_update(
    test_db: Any,
) -> None:
    """Default (for_update) strategy takes FOR UPDATE and passes the budget check."""
    budget = Budget(
        budget_id="race-budget",
        max_budget=10.0,
    )
    test_db.add(budget)

    user = User(
        user_id="race-user",
        spend=9.99,
        budget_id="race-budget",
    )
    test_db.add(user)
    await test_db.commit()

    result = await validate_user_budget(test_db, "race-user")
    assert result.user_id == "race-user"


@pytest.mark.asyncio
async def test_budget_check_rejects_at_limit(
    test_db: Any,
) -> None:
    """Test that a user at or over budget limit is rejected."""
    budget = Budget(
        budget_id="full-budget",
        max_budget=10.0,
    )
    test_db.add(budget)

    user = User(
        user_id="full-user",
        spend=10.0,
        budget_id="full-budget",
    )
    test_db.add(user)
    await test_db.commit()

    with pytest.raises(HTTPException) as exc_info:
        await validate_user_budget(test_db, "full-user")
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_cas_strategy_takes_no_for_update_lock(
    test_db: Any,
) -> None:
    """cas strategy must never call get_active_user with for_update=True."""
    budget = Budget(budget_id="cas-budget", max_budget=10.0, budget_duration_sec=3600)
    test_db.add(budget)

    user = User(
        user_id="cas-user",
        spend=1.0,
        budget_id="cas-budget",
        next_budget_reset_at=datetime.now(UTC) + timedelta(hours=1),
    )
    test_db.add(user)
    await test_db.commit()

    with patch.object(budget_service, "get_active_user", wraps=budget_service.get_active_user) as spy:
        result = await validate_user_budget(test_db, "cas-user", strategy="cas")

    assert result.user_id == "cas-user"
    for call in spy.call_args_list:
        assert call.kwargs.get("for_update") is False, (
            f"cas strategy must never take FOR UPDATE, got {call.kwargs}"
        )


@pytest.mark.asyncio
async def test_cas_strategy_resets_due_budget(
    test_db: Any,
) -> None:
    """cas strategy resets an overdue budget via atomic UPDATE without FOR UPDATE."""
    budget = Budget(budget_id="cas-reset-budget", max_budget=10.0, budget_duration_sec=3600)
    test_db.add(budget)

    user = User(
        user_id="cas-reset-user",
        spend=5.0,
        budget_id="cas-reset-budget",
        next_budget_reset_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    test_db.add(user)
    await test_db.commit()

    with patch.object(budget_service, "get_active_user", wraps=budget_service.get_active_user) as spy:
        result = await validate_user_budget(test_db, "cas-reset-user", strategy="cas")

    assert result.user_id == "cas-reset-user"
    assert result.spend == 0.0
    for call in spy.call_args_list:
        assert call.kwargs.get("for_update") is False, (
            f"cas strategy must never take FOR UPDATE even on reset, got {call.kwargs}"
        )


@pytest.mark.asyncio
async def test_disabled_strategy_skips_validation(
    test_db: Any,
) -> None:
    """disabled strategy returns None without running any DB checks."""
    with patch.object(budget_service, "get_active_user", wraps=budget_service.get_active_user) as spy:
        result = await validate_user_budget(test_db, "nonexistent-user", strategy="disabled")

    assert result is None
    assert spy.call_count == 0, "disabled strategy must not touch the DB"
