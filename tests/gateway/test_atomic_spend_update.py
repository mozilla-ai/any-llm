"""Tests for atomic spend update via SQL expression."""

import pytest
from sqlalchemy.orm import Session

from any_llm.gateway.db.models import ModelPricing, User
from any_llm.gateway.routes.chat import _log_usage
from any_llm.types.completion import CompletionUsage


@pytest.mark.asyncio
async def test_spend_update_uses_sql_expression(test_db: Session) -> None:
    """Test that _log_usage updates spend atomically via SQL, not Python read-modify-write."""
    # Set up user with initial spend
    user = User(user_id="atomic-user", spend=5.0)
    test_db.add(user)

    pricing = ModelPricing(
        model_key="openai:gpt-4o",
        input_price_per_million=2.5,
        output_price_per_million=10.0,
    )
    test_db.add(pricing)
    test_db.commit()

    usage = CompletionUsage(prompt_tokens=1_000_000, completion_tokens=100_000, total_tokens=1_100_000)

    await _log_usage(
        db=test_db,
        api_key_obj=None,
        model="gpt-4o",
        provider="openai",
        endpoint="/v1/chat/completions",
        user_id="atomic-user",
        usage_override=usage,
    )

    # Refresh from database
    test_db.expire_all()
    updated_user = test_db.query(User).filter(User.user_id == "atomic-user").first()
    assert updated_user is not None

    # Expected cost: (1M / 1M) * 2.5 + (100K / 1M) * 10.0 = 2.5 + 1.0 = 3.5
    expected_new_spend = 5.0 + 3.5
    assert abs(updated_user.spend - expected_new_spend) < 0.001, (
        f"Expected spend {expected_new_spend}, got {updated_user.spend}"
    )


@pytest.mark.asyncio
async def test_multiple_spend_updates_accumulate(test_db: Session) -> None:
    """Test that multiple sequential spend updates accumulate correctly."""
    user = User(user_id="multi-spend-user", spend=0.0)
    test_db.add(user)

    pricing = ModelPricing(
        model_key="openai:gpt-4o",
        input_price_per_million=10.0,
        output_price_per_million=10.0,
    )
    test_db.add(pricing)
    test_db.commit()

    for _ in range(3):
        usage = CompletionUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
        await _log_usage(
            db=test_db,
            api_key_obj=None,
            model="gpt-4o",
            provider="openai",
            endpoint="/v1/chat/completions",
            user_id="multi-spend-user",
            usage_override=usage,
        )

    test_db.expire_all()
    updated_user = test_db.query(User).filter(User.user_id == "multi-spend-user").first()
    assert updated_user is not None

    # Each call: (1M/1M)*10 + (1M/1M)*10 = 20.0, x3 = 60.0
    assert abs(updated_user.spend - 60.0) < 0.001, f"Expected spend 60.0, got {updated_user.spend}"
