from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import AnyLLMError
from any_llm.gateway.db import Budget, BudgetResetLog, User
from any_llm.gateway.log_config import logger
from any_llm.gateway.metrics import record_budget_exceeded
from any_llm.gateway.pricing import find_model_pricing


def calculate_next_reset(start: datetime, duration_sec: int) -> datetime:
    """Calculate next budget reset datetime.

    Args:
        start: Starting datetime for the budget period
        duration_sec: Duration in seconds

    Returns:
        datetime when the budget should next reset

    """
    return start + timedelta(seconds=duration_sec)


def reset_user_budget(db: Session, user: User, budget: Budget, now: datetime) -> None:
    """Reset user's budget spend and schedule next reset.

    Args:
        db: Database session
        user: User object to reset
        budget: Budget object associated with user
        now: Current timestamp to use for reset timing

    """
    previous_spend = user.spend

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

    try:
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("Failed to commit budget reset for user '%s': %s", user.user_id, e)
        raise


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
    user = db.query(User).filter(User.user_id == user_id, User.deleted_at.is_(None)).with_for_update().first()

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
                reset_user_budget(db, user, budget, now)

            if budget.max_budget is not None:
                if user.spend >= budget.max_budget:
                    if model and _is_model_free(db, model):
                        return user
                    record_budget_exceeded(user_id)
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
    try:
        provider, model_name = AnyLLM.split_model_provider(model)
        provider_str = provider.value if provider else None
        pricing = find_model_pricing(db, provider_str, model_name)
        if pricing:
            return pricing.input_price_per_million == 0 and pricing.output_price_per_million == 0
    except (AnyLLMError, ValueError, SQLAlchemyError) as e:
        logger.warning("Failed to determine provider pricing: %s", e)

    return False
