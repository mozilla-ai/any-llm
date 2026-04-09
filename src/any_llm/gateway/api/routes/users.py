from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from any_llm.gateway.api.deps import get_db, verify_master_key
from any_llm.gateway.models.entities import APIKey, Budget, UsageLog, User
from any_llm.gateway.repositories.users_repository import get_active_user
from any_llm.gateway.services.budget_service import calculate_next_reset

router = APIRouter(prefix="/v1/users", tags=["users"])


class CreateUserRequest(BaseModel):
    """Request model for creating a new user."""

    user_id: str = Field(description="Unique user identifier")
    alias: str | None = Field(default=None, description="Optional admin-facing alias")
    budget_id: str | None = Field(default=None, description="Optional budget ID")
    blocked: bool = Field(default=False, description="Whether user is blocked")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class UserResponse(BaseModel):
    """Response model for user information."""

    user_id: str
    alias: str | None
    spend: float
    budget_id: str | None
    budget_started_at: str | None
    next_budget_reset_at: str | None
    blocked: bool
    created_at: str
    updated_at: str
    metadata: dict[str, Any]

    @classmethod
    def from_model(cls, user: User) -> "UserResponse":
        return cls(
            user_id=user.user_id,
            alias=user.alias,
            spend=float(user.spend),
            budget_id=user.budget_id,
            budget_started_at=user.budget_started_at.isoformat() if user.budget_started_at else None,
            next_budget_reset_at=user.next_budget_reset_at.isoformat() if user.next_budget_reset_at else None,
            blocked=bool(user.blocked),
            created_at=user.created_at.isoformat(),
            updated_at=user.updated_at.isoformat(),
            metadata=dict(user.metadata_) if user.metadata_ else {},
        )


class UpdateUserRequest(BaseModel):
    """Request model for updating a user."""

    alias: str | None = None
    budget_id: str | None = None
    blocked: bool | None = None
    metadata: dict[str, Any] | None = None


class UsageLogResponse(BaseModel):
    """Response model for usage log."""

    id: str
    user_id: str | None
    api_key_id: str | None
    timestamp: str
    model: str
    provider: str | None
    endpoint: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost: float | None
    status: str
    error_message: str | None

    @classmethod
    def from_model(cls, log: UsageLog) -> "UsageLogResponse":
        return cls(
            id=log.id,
            user_id=log.user_id,
            api_key_id=log.api_key_id,
            timestamp=log.timestamp.isoformat(),
            model=log.model,
            provider=log.provider,
            endpoint=log.endpoint,
            prompt_tokens=log.prompt_tokens,
            completion_tokens=log.completion_tokens,
            total_tokens=log.total_tokens,
            cost=log.cost,
            status=log.status,
            error_message=log.error_message,
        )


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_user(
    request: CreateUserRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    """Create a new user."""
    existing_user = (await db.execute(select(User).where(User.user_id == request.user_id))).scalar_one_or_none()
    if existing_user and existing_user.deleted_at is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with id '{request.user_id}' already exists",
        )

    if existing_user and existing_user.deleted_at is not None:
        user = existing_user
        user.deleted_at = None
        user.spend = 0.0
        user.alias = request.alias
        user.budget_id = request.budget_id
        user.blocked = request.blocked
        user.metadata_ = request.metadata
        user.budget_started_at = None
        user.next_budget_reset_at = None
    else:
        user = User(
            user_id=request.user_id,
            alias=request.alias,
            budget_id=request.budget_id,
            blocked=request.blocked,
            metadata_=request.metadata,
        )
        db.add(user)

    if request.budget_id:
        budget = (await db.execute(select(Budget).where(Budget.budget_id == request.budget_id))).scalar_one_or_none()
        if not budget:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Budget with id '{request.budget_id}' not found",
            )

        now = datetime.now(UTC)
        user.budget_started_at = now
        if budget.budget_duration_sec:
            user.next_budget_reset_at = calculate_next_reset(now, budget.budget_duration_sec)

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(user)

    return UserResponse.from_model(user)


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_users(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[UserResponse]:
    """List all users with pagination."""
    users = (
        (await db.execute(select(User).where(User.deleted_at.is_(None)).offset(skip).limit(limit))).scalars().all()
    )

    return [UserResponse.from_model(user) for user in users]


@router.get("/{user_id}", dependencies=[Depends(verify_master_key)])
async def get_user(
    user_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    """Get details of a specific user."""
    user = await get_active_user(db, user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id '{user_id}' not found",
        )

    return UserResponse.from_model(user)


@router.patch("/{user_id}", dependencies=[Depends(verify_master_key)])
async def update_user(
    user_id: str,
    request: UpdateUserRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    """Update a user."""
    user = await get_active_user(db, user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id '{user_id}' not found",
        )

    if request.alias is not None:
        user.alias = request.alias
    if request.budget_id is not None:
        budget = (await db.execute(select(Budget).where(Budget.budget_id == request.budget_id))).scalar_one_or_none()
        if not budget:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Budget with id '{request.budget_id}' not found",
            )

        user.budget_id = request.budget_id
        now = datetime.now(UTC)
        user.budget_started_at = now
        if budget.budget_duration_sec:
            user.next_budget_reset_at = calculate_next_reset(now, budget.budget_duration_sec)
        else:
            user.next_budget_reset_at = None
    if request.blocked is not None:
        user.blocked = request.blocked
    if request.metadata is not None:
        user.metadata_ = request.metadata

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(user)

    return UserResponse.from_model(user)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_user(
    user_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a user."""
    user = await get_active_user(db, user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id '{user_id}' not found",
        )

    await db.execute(update(APIKey).where(APIKey.user_id == user_id).values(is_active=False))
    user.deleted_at = datetime.now(UTC)

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None


@router.get("/{user_id}/usage", dependencies=[Depends(verify_master_key)])
async def get_user_usage(
    user_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[UsageLogResponse]:
    """Get usage history for a specific user."""
    user = (await db.execute(select(User).where(User.user_id == user_id))).scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id '{user_id}' not found",
        )

    usage_logs = (
        (
            await db.execute(
                select(UsageLog)
                .where(UsageLog.user_id == user_id)
                .order_by(UsageLog.timestamp.desc())
                .offset(skip)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )

    return [UsageLogResponse.from_model(log) for log in usage_logs]
