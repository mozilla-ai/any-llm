from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from any_llm.gateway.auth import verify_master_key
from any_llm.gateway.budget import calculate_next_reset
from any_llm.gateway.db import Budget, UsageLog, User, get_db

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


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_user(
    request: CreateUserRequest,
    db: Annotated[Session, Depends(get_db)],
) -> UserResponse:
    """Create a new user."""
    existing_user = db.query(User).filter(User.user_id == request.user_id).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with id '{request.user_id}' already exists",
        )

    user = User(
        user_id=request.user_id,
        alias=request.alias,
        budget_id=request.budget_id,
        blocked=request.blocked,
        metadata_=request.metadata,
    )

    if request.budget_id:
        budget = db.query(Budget).filter(Budget.budget_id == request.budget_id).first()
        if not budget:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Budget with id '{request.budget_id}' not found",
            )

        now = datetime.now(UTC)
        user.budget_started_at = now
        if budget.budget_duration_sec:
            user.next_budget_reset_at = calculate_next_reset(now, budget.budget_duration_sec)

    db.add(user)
    db.commit()
    db.refresh(user)

    return UserResponse(
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


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_users(
    db: Annotated[Session, Depends(get_db)],
    skip: int = 0,
    limit: int = 100,
) -> list[UserResponse]:
    """List all users with pagination."""
    users = db.query(User).offset(skip).limit(limit).all()

    return [
        UserResponse(
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
        for user in users
    ]


@router.get("/{user_id}", dependencies=[Depends(verify_master_key)])
async def get_user(
    user_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> UserResponse:
    """Get details of a specific user."""
    user = db.query(User).filter(User.user_id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id '{user_id}' not found",
        )

    return UserResponse(
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


@router.patch("/{user_id}", dependencies=[Depends(verify_master_key)])
async def update_user(
    user_id: str,
    request: UpdateUserRequest,
    db: Annotated[Session, Depends(get_db)],
) -> UserResponse:
    """Update a user."""
    user = db.query(User).filter(User.user_id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id '{user_id}' not found",
        )

    if request.alias is not None:
        user.alias = request.alias
    if request.budget_id is not None:
        budget = db.query(Budget).filter(Budget.budget_id == request.budget_id).first()
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

    db.commit()
    db.refresh(user)

    return UserResponse(
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


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_user(
    user_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Delete a user."""
    user = db.query(User).filter(User.user_id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id '{user_id}' not found",
        )

    db.delete(user)
    db.commit()


@router.get("/{user_id}/usage", dependencies=[Depends(verify_master_key)])
async def get_user_usage(
    user_id: str,
    db: Annotated[Session, Depends(get_db)],
    skip: int = 0,
    limit: int = 100,
) -> list[UsageLogResponse]:
    """Get usage history for a specific user."""
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id '{user_id}' not found",
        )

    usage_logs = (
        db.query(UsageLog)
        .filter(UsageLog.user_id == user_id)
        .order_by(UsageLog.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return [
        UsageLogResponse(
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
        for log in usage_logs
    ]
