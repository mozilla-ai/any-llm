from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from any_llm.any_llm import AnyLLM
from any_llm.gateway.api.deps import get_db, verify_master_key
from any_llm.gateway.models.entities import ModelPricing

router = APIRouter(prefix="/v1/pricing", tags=["pricing"])


class SetPricingRequest(BaseModel):
    """Request model for setting model pricing."""

    model_key: str = Field(description="Model identifier in format 'provider:model'")
    input_price_per_million: float = Field(ge=0, description="Price per 1M input tokens")
    output_price_per_million: float = Field(ge=0, description="Price per 1M output tokens")
    effective_at: datetime | None = Field(
        default=None,
        description="ISO 8601 datetime from which this price applies. Defaults to now if omitted.",
    )


class PricingResponse(BaseModel):
    """Response model for model pricing."""

    model_key: str
    effective_at: str
    input_price_per_million: float
    output_price_per_million: float
    created_at: str
    updated_at: str

    @classmethod
    def from_model(cls, pricing: "ModelPricing") -> "PricingResponse":
        """Create a PricingResponse from a ModelPricing ORM model."""
        return cls(
            model_key=pricing.model_key,
            effective_at=pricing.effective_at.isoformat(),
            input_price_per_million=pricing.input_price_per_million,
            output_price_per_million=pricing.output_price_per_million,
            created_at=pricing.created_at.isoformat(),
            updated_at=pricing.updated_at.isoformat(),
        )


@router.post("", dependencies=[Depends(verify_master_key)])
async def set_pricing(
    request: SetPricingRequest,
    db: Annotated[Session, Depends(get_db)],
) -> PricingResponse:
    """Set or update pricing for a model at a specific effective date.

    If a price entry with the same (model_key, effective_at) exists, it is
    updated. Otherwise a new entry is created.
    """
    provider, model_name = AnyLLM.split_model_provider(request.model_key)
    normalized_key = f"{provider.value}:{model_name}"
    effective_at = request.effective_at or datetime.now(UTC)

    pricing = (
        db.query(ModelPricing)
        .filter(
            ModelPricing.model_key == normalized_key,
            ModelPricing.effective_at == effective_at,
        )
        .first()
    )

    if pricing:
        pricing.input_price_per_million = request.input_price_per_million
        pricing.output_price_per_million = request.output_price_per_million
    else:
        pricing = ModelPricing(
            model_key=normalized_key,
            effective_at=effective_at,
            input_price_per_million=request.input_price_per_million,
            output_price_per_million=request.output_price_per_million,
        )
        db.add(pricing)

    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    db.refresh(pricing)

    return PricingResponse.from_model(pricing)


@router.get("")
async def list_pricing(
    db: Annotated[Session, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[PricingResponse]:
    """List all model pricing entries (all effective dates)."""
    pricings = (
        db.query(ModelPricing)
        .order_by(ModelPricing.model_key, ModelPricing.effective_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return [PricingResponse.from_model(pricing) for pricing in pricings]


@router.get("/{model_key:path}/history")
async def get_pricing_history(
    model_key: str,
    db: Annotated[Session, Depends(get_db)],
) -> list[PricingResponse]:
    """Get all pricing entries for a specific model, ordered newest first."""
    pricings = (
        db.query(ModelPricing)
        .filter(ModelPricing.model_key == model_key)
        .order_by(ModelPricing.effective_at.desc())
        .all()
    )

    if not pricings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pricing found for model '{model_key}'",
        )

    return [PricingResponse.from_model(p) for p in pricings]


@router.get("/{model_key:path}")
async def get_pricing(
    model_key: str,
    db: Annotated[Session, Depends(get_db)],
    as_of: datetime | None = Query(default=None, description="Return the price effective at this timestamp"),
) -> PricingResponse:
    """Get the effective pricing for a specific model.

    By default returns the latest price. Pass as_of to get the price
    that was in effect at a specific point in time.
    """
    query_time = as_of or datetime.now(UTC)

    pricing = (
        db.query(ModelPricing)
        .filter(ModelPricing.model_key == model_key, ModelPricing.effective_at <= query_time)
        .order_by(ModelPricing.effective_at.desc())
        .first()
    )

    if not pricing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pricing for model '{model_key}' not found",
        )

    return PricingResponse.from_model(pricing)


@router.delete("/{model_key:path}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_pricing(
    model_key: str,
    db: Annotated[Session, Depends(get_db)],
    effective_at: datetime | None = Query(
        default=None,
        description="Delete only the entry at this effective date. If omitted, deletes all entries for the model.",
    ),
) -> None:
    """Delete pricing for a model.

    If effective_at is provided, deletes only the specific price entry.
    Otherwise deletes all price entries for the model.
    """
    query = db.query(ModelPricing).filter(ModelPricing.model_key == model_key)
    if effective_at is not None:
        query = query.filter(ModelPricing.effective_at == effective_at)

    count = query.count()
    if count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pricing for model '{model_key}' not found",
        )

    query.delete()
    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
