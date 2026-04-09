from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from any_llm.gateway.api.deps import get_db, verify_master_key
from any_llm.gateway.models.entities import Budget

router = APIRouter(prefix="/v1/budgets", tags=["budgets"])


class CreateBudgetRequest(BaseModel):
    """Request model for creating a new budget."""

    max_budget: float | None = Field(default=None, ge=0, description="Maximum spending limit")
    budget_duration_sec: int | None = Field(
        default=None, gt=0, description="Budget duration in seconds (e.g., 86400 for daily, 604800 for weekly)"
    )


class BudgetResponse(BaseModel):
    """Response model for budget information."""

    budget_id: str
    max_budget: float | None
    budget_duration_sec: int | None
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, budget: "Budget") -> "BudgetResponse":
        """Create a BudgetResponse from a Budget ORM model."""
        return cls(
            budget_id=budget.budget_id,
            max_budget=budget.max_budget,
            budget_duration_sec=budget.budget_duration_sec,
            created_at=budget.created_at.isoformat(),
            updated_at=budget.updated_at.isoformat(),
        )


class UpdateBudgetRequest(BaseModel):
    """Request model for updating a budget."""

    max_budget: float | None = Field(default=None, ge=0)
    budget_duration_sec: int | None = Field(default=None, gt=0)


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_budget(
    request: CreateBudgetRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetResponse:
    """Create a new budget."""
    budget = Budget(
        max_budget=request.max_budget,
        budget_duration_sec=request.budget_duration_sec,
    )

    db.add(budget)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(budget)

    return BudgetResponse.from_model(budget)


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_budgets(
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[BudgetResponse]:
    """List all budgets with pagination."""
    budgets = (await db.execute(select(Budget).offset(skip).limit(limit))).scalars().all()

    return [BudgetResponse.from_model(budget) for budget in budgets]


@router.get("/{budget_id}", dependencies=[Depends(verify_master_key)])
async def get_budget(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetResponse:
    """Get details of a specific budget."""
    budget = (await db.execute(select(Budget).where(Budget.budget_id == budget_id))).scalar_one_or_none()

    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )

    return BudgetResponse.from_model(budget)


@router.patch("/{budget_id}", dependencies=[Depends(verify_master_key)])
async def update_budget(
    budget_id: str,
    request: UpdateBudgetRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BudgetResponse:
    """Update a budget."""
    budget = (await db.execute(select(Budget).where(Budget.budget_id == budget_id))).scalar_one_or_none()

    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )

    if request.max_budget is not None:
        budget.max_budget = request.max_budget
    if request.budget_duration_sec is not None:
        budget.budget_duration_sec = request.budget_duration_sec

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(budget)

    return BudgetResponse.from_model(budget)


@router.delete("/{budget_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_budget(
    budget_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete a budget."""
    budget = (await db.execute(select(Budget).where(Budget.budget_id == budget_id))).scalar_one_or_none()

    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )

    await db.delete(budget)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
