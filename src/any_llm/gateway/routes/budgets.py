from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from any_llm.gateway.auth import verify_master_key
from any_llm.gateway.db import Budget, get_db

router = APIRouter(prefix="/v1/budgets", tags=["budgets"])


class CreateBudgetRequest(BaseModel):
    """Request model for creating a new budget."""

    max_budget: float | None = Field(default=None, description="Maximum spending limit")
    budget_duration_sec: int | None = Field(
        default=None, description="Budget duration in seconds (e.g., 86400 for daily, 604800 for weekly)"
    )


class BudgetResponse(BaseModel):
    """Response model for budget information."""

    budget_id: str
    max_budget: float | None
    budget_duration_sec: int | None
    created_at: str
    updated_at: str


class UpdateBudgetRequest(BaseModel):
    """Request model for updating a budget."""

    max_budget: float | None = None
    budget_duration_sec: int | None = None


@router.post("", dependencies=[Depends(verify_master_key)])
async def create_budget(
    request: CreateBudgetRequest,
    db: Annotated[Session, Depends(get_db)],
) -> BudgetResponse:
    """Create a new budget."""
    budget = Budget(
        max_budget=request.max_budget,
        budget_duration_sec=request.budget_duration_sec,
    )

    db.add(budget)
    db.commit()
    db.refresh(budget)

    return BudgetResponse(
        budget_id=budget.budget_id,
        max_budget=budget.max_budget,
        budget_duration_sec=budget.budget_duration_sec,
        created_at=budget.created_at.isoformat(),
        updated_at=budget.updated_at.isoformat(),
    )


@router.get("", dependencies=[Depends(verify_master_key)])
async def list_budgets(
    db: Annotated[Session, Depends(get_db)],
    skip: int = 0,
    limit: int = 100,
) -> list[BudgetResponse]:
    """List all budgets with pagination."""
    budgets = db.query(Budget).offset(skip).limit(limit).all()

    return [
        BudgetResponse(
            budget_id=budget.budget_id,
            max_budget=budget.max_budget,
            budget_duration_sec=budget.budget_duration_sec,
            created_at=budget.created_at.isoformat(),
            updated_at=budget.updated_at.isoformat(),
        )
        for budget in budgets
    ]


@router.get("/{budget_id}", dependencies=[Depends(verify_master_key)])
async def get_budget(
    budget_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> BudgetResponse:
    """Get details of a specific budget."""
    budget = db.query(Budget).filter(Budget.budget_id == budget_id).first()

    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )

    return BudgetResponse(
        budget_id=budget.budget_id,
        max_budget=budget.max_budget,
        budget_duration_sec=budget.budget_duration_sec,
        created_at=budget.created_at.isoformat(),
        updated_at=budget.updated_at.isoformat(),
    )


@router.patch("/{budget_id}", dependencies=[Depends(verify_master_key)])
async def update_budget(
    budget_id: str,
    request: UpdateBudgetRequest,
    db: Annotated[Session, Depends(get_db)],
) -> BudgetResponse:
    """Update a budget."""
    budget = db.query(Budget).filter(Budget.budget_id == budget_id).first()

    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )

    if request.max_budget is not None:
        budget.max_budget = request.max_budget
    if request.budget_duration_sec is not None:
        budget.budget_duration_sec = request.budget_duration_sec

    db.commit()
    db.refresh(budget)

    return BudgetResponse(
        budget_id=budget.budget_id,
        max_budget=budget.max_budget,
        budget_duration_sec=budget.budget_duration_sec,
        created_at=budget.created_at.isoformat(),
        updated_at=budget.updated_at.isoformat(),
    )


@router.delete("/{budget_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_budget(
    budget_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Delete a budget."""
    budget = db.query(Budget).filter(Budget.budget_id == budget_id).first()

    if not budget:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Budget with id '{budget_id}' not found",
        )

    db.delete(budget)
    db.commit()
