from datetime import UTC, datetime, timedelta

from fastapi import HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import AnyLLMError
from any_llm.gateway.log_config import logger
from any_llm.gateway.metrics import record_budget_exceeded
from any_llm.gateway.models.entities import Budget, BudgetResetLog, User
from any_llm.gateway.repositories.users_repository import get_active_user
from any_llm.gateway.services.pricing_service import find_model_pricing


def calculate_next_reset(start: datetime, duration_sec: int) -> datetime:
    """Calculate next budget reset datetime."""
    return start + timedelta(seconds=duration_sec)


async def reset_user_budget(db: AsyncSession, user: User, budget: Budget, now: datetime) -> None:
    """Reset user's budget spend and schedule next reset (ORM path).

    Assumes the caller holds an appropriate serialization guarantee on the user
    row (e.g. via FOR UPDATE in the 'for_update' strategy). The 'cas' strategy
    uses an atomic conditional UPDATE instead and does not call this function.
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
        await db.commit()
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error("Failed to commit budget reset for user '%s': %s", user.user_id, e)
        raise


async def _claim_reset_cas(db: AsyncSession, user: User, budget: Budget, now: datetime) -> bool:
    """Attempt to reset the user's budget via an atomic conditional UPDATE.

    Returns True if this caller won the reset race (and inserted the BudgetResetLog),
    False if another caller already reset the row. No explicit FOR UPDATE lock is taken.
    """
    previous_spend = user.spend
    next_reset = calculate_next_reset(now, budget.budget_duration_sec) if budget.budget_duration_sec else None

    # Compare-and-swap: the WHERE clause gates the write on next_budget_reset_at still
    # being overdue. If another worker already reset the row, rowcount == 0 and we no-op.
    result = await db.execute(
        update(User)
        .where(
            User.user_id == user.user_id,
            User.deleted_at.is_(None),
            User.next_budget_reset_at < now,
        )
        .values(
            spend=0.0,
            budget_started_at=now,
            next_budget_reset_at=next_reset,
        )
    )
    if result.rowcount == 0:
        await db.commit()
        return False

    # We claimed the reset. Record it and refresh the ORM-attached user.
    db.add(
        BudgetResetLog(
            user_id=user.user_id,
            budget_id=budget.budget_id,
            previous_spend=previous_spend,
            reset_at=now,
            next_reset_at=next_reset,
        )
    )
    user.spend = 0.0
    user.budget_started_at = now
    user.next_budget_reset_at = next_reset

    try:
        await db.commit()
    except SQLAlchemyError as e:
        await db.rollback()
        logger.error("Failed to commit CAS budget reset for user '%s': %s", user.user_id, e)
        raise
    return True


async def validate_user_budget(
    db: AsyncSession,
    user_id: str,
    model: str | None = None,
    *,
    strategy: str = "for_update",
) -> User | None:
    """Validate user exists, is not blocked, and has available budget.

    Args:
        db: Async database session
        user_id: User identifier
        model: Optional model identifier for free-model bypass
        strategy: 'for_update' (default, legacy — FOR UPDATE across entire request),
            'cas' (recommended — lock-free conditional UPDATE), or
            'disabled' (no validation at all — returns None).

    Raises:
        HTTPException: If user is blocked, doesn't exist, or exceeded budget.
    """
    match strategy:
        case "disabled":
            return None
        case "cas":
            return await _validate_cas(db, user_id, model)
        case "for_update":
            return await _validate_for_update(db, user_id, model)
        case _:
            logger.warning("Unrecognized budget_strategy '%s', falling back to 'for_update'", strategy)
            return await _validate_for_update(db, user_id, model)


async def _validate_cas(db: AsyncSession, user_id: str, model: str | None) -> User:
    """Read unlocked; do the reset via atomic conditional UPDATE (no FOR UPDATE)."""
    user = await get_active_user(db, user_id, for_update=False)
    await db.commit()

    _check_user_active(user, user_id)
    assert user is not None  # noqa: S101

    if not user.budget_id:
        return user

    budget = (await db.execute(select(Budget).where(Budget.budget_id == user.budget_id))).scalar_one_or_none()
    await db.commit()
    if not budget:
        return user

    now = datetime.now(UTC)
    if user.next_budget_reset_at and now >= user.next_budget_reset_at:
        won_reset = await _claim_reset_cas(db, user, budget, now)
        if not won_reset:
            # Another request won the CAS race and reset the budget.
            # Refresh so we see the post-reset spend (0.0) instead of
            # the stale pre-reset value, which could cause a false 403.
            await db.refresh(user)

    await _enforce_budget_limit(db, user, budget, model, user_id)
    return user


async def _validate_for_update(db: AsyncSession, user_id: str, model: str | None) -> User:
    """Legacy: hold FOR UPDATE across entire request. Provided for benchmark/rollback."""
    user = await get_active_user(db, user_id, for_update=True)

    _check_user_active(user, user_id)
    assert user is not None  # noqa: S101

    if user.budget_id:
        budget = (await db.execute(select(Budget).where(Budget.budget_id == user.budget_id))).scalar_one_or_none()
        if budget:
            now = datetime.now(UTC)
            if user.next_budget_reset_at and now >= user.next_budget_reset_at:
                await reset_user_budget(db, user, budget, now)

            await _enforce_budget_limit(db, user, budget, model, user_id)

    return user


def _check_user_active(user: User | None, user_id: str) -> None:
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


async def _enforce_budget_limit(
    db: AsyncSession, user: User, budget: Budget, model: str | None, user_id: str
) -> None:
    if budget.max_budget is not None and user.spend >= budget.max_budget:
        if model and await _is_model_free(db, model):
            return
        record_budget_exceeded(user_id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User '{user_id}' has exceeded budget limit",
        )


async def _is_model_free(db: AsyncSession, model: str) -> bool:
    """Check if a model is free (both input and output prices are 0)."""
    try:
        provider, model_name = AnyLLM.split_model_provider(model)
        provider_str = provider.value if provider else None
        pricing = await find_model_pricing(db, provider_str, model_name)
        if pricing:
            return pricing.input_price_per_million == 0 and pricing.output_price_per_million == 0
    except (AnyLLMError, ValueError, SQLAlchemyError) as e:
        logger.warning("Failed to determine provider pricing: %s", e)

    return False
