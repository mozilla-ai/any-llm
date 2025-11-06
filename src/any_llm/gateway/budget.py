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


async def validate_user_budget(db: Session, user_id: str) -> User:
    """Validate user exists, is not blocked, and has available budget.

    Args:
        db: Database session
        user_id: User identifier

    Returns:
        User object if validation passes

    Raises:
        HTTPException: If user is blocked, doesn't exist, or exceeded budget

    """
    user = db.query(User).filter(User.user_id == user_id).first()

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
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"User '{user_id}' has exceeded budget limit",
                    )

    return user
