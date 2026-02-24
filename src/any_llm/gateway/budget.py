from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from any_llm.gateway.db import Budget, BudgetResetLog, User


def calculate_next_reset(start: datetime, duration_sec: int) -> datetime:
    """Calculate next budget reset datetime.

    Args:
        start: Starting datetime for the budget period
        duration_sec: Duration in seconds

    Returns:
        datetime when the budget should next reset

    """
    return start + timedelta(seconds=duration_sec)


def reset_user_budget(db: Session, user: User, budget: Budget) -> None:
    """Reset user's budget spend and schedule next reset.

    Args:
        db: Database session
        user: User object to reset
        budget: Budget object associated with user

    """
    previous_spend = user.spend
    now = datetime.now(UTC)

    user.spend = 0.0
    user.budget_started_at = now

    if budget.budget_duration_sec:
        user.next_budget_reset_at = calculate_next_reset(now, budget.budget_duration_sec)
    else:
        user.next_budget_reset_at = None

    reset_log = BudgetResetLog(
        user_id=user.user_id,
        budget_id=budget.budget_id,
        previous_spend=previous_spend,
        reset_at=now,
        next_reset_at=user.next_budget_reset_at,
    )
    db.add(reset_log)
    db.commit()


async def validate_user_budget(db: Session, user_id: str, model: str | None = None) -> User:
    """Validate user exists, is not blocked, and has available budget.

    Args:
        db: Database session
        user_id: User identifier
        model: Optional model identifier (e.g., "provider/model") to check if it's a free model

    Returns:
        User object if validation passes

    Raises:
        HTTPException: If user is blocked, doesn't exist, or exceeded budget

    """
    user = db.query(User).filter(User.user_id == user_id).with_for_update().first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User '{user_id}' not found",
        )

    if user.blocked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user_id}' is blocked",
        )

    if user.budget_id:
        budget = db.query(Budget).filter(Budget.budget_id == user.budget_id).first()
        if budget:
            now = datetime.now(UTC)
            if user.next_budget_reset_at and now >= user.next_budget_reset_at:
                reset_user_budget(db, user, budget)

            if budget.max_budget is not None:
                if user.spend >= budget.max_budget:
                    if model and _is_model_free(db, model):
                        return user
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"User '{user_id}' has exceeded budget limit",
                    )

    return user


def _is_model_free(db: Session, model: str) -> bool:
    """Check if a model is free (both input and output prices are 0).

    Args:
        db: Database session
        model: Model identifier (e.g., "provider/model" or "model")

    Returns:
        True if the model is free, False otherwise or if pricing not found

    """
    from any_llm.gateway.db import ModelPricing

    provider, model_name = _split_model_provider(model)
    model_key = f"{provider}:{model_name}" if provider else model_name
    model_key_legacy = f"{provider}/{model_name}" if provider else None

    pricing = db.query(ModelPricing).filter(ModelPricing.model_key == model_key).first()
    if not pricing and model_key_legacy:
        pricing = db.query(ModelPricing).filter(ModelPricing.model_key == model_key_legacy).first()

    if pricing:
        return pricing.input_price_per_million == 0 and pricing.output_price_per_million == 0

    return False


def _split_model_provider(model: str) -> tuple[str | None, str]:
    """Split model identifier into provider and model name.

    Args:
        model: Model identifier (e.g., "openai/gpt-4o" or "gpt-4o")

    Returns:
        Tuple of (provider, model_name)

    """
    if "/" in model:
        parts = model.split("/", 1)
        return parts[0], parts[1]
    return None, model
