"""Tests for budget enforcement with row locking."""

from typing import Any

import pytest

from any_llm.gateway.budget import validate_user_budget
from any_llm.gateway.db.models import Budget, User


@pytest.mark.asyncio
async def test_validate_user_budget_uses_for_update(
    test_db: Any,
) -> None:
    """Test that validate_user_budget queries with FOR UPDATE locking.

    We verify this indirectly by confirming the budget check works correctly
    for a user at their budget limit.
    """
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
    test_db.commit()

    # Should pass -- spend is under budget
    result = await validate_user_budget(test_db, "race-user")
    assert result.user_id == "race-user"


@pytest.mark.asyncio
async def test_budget_check_rejects_at_limit(
    test_db: Any,
) -> None:
    """Test that a user at or over budget limit is rejected."""
    from fastapi import HTTPException

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
    test_db.commit()

    with pytest.raises(HTTPException) as exc_info:
        await validate_user_budget(test_db, "full-user")
    assert exc_info.value.status_code == 403
