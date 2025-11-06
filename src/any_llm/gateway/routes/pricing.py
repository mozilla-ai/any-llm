from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from any_llm.gateway.auth import verify_master_key
from any_llm.gateway.db import ModelPricing, get_db

router = APIRouter(prefix="/v1/pricing", tags=["pricing"])


class SetPricingRequest(BaseModel):
    """Request model for setting model pricing."""

    model_key: str = Field(description="Model identifier in format 'provider:model'")
    input_price_per_million: float = Field(description="Price per 1M input tokens")
    output_price_per_million: float = Field(description="Price per 1M output tokens")


class PricingResponse(BaseModel):
    """Response model for model pricing."""

    model_key: str
    input_price_per_million: float
    output_price_per_million: float
    created_at: str
    updated_at: str


@router.post("", dependencies=[Depends(verify_master_key)])
async def set_pricing(
    request: SetPricingRequest,
    db: Annotated[Session, Depends(get_db)],
) -> PricingResponse:
    """Set or update pricing for a model."""
    pricing = db.query(ModelPricing).filter(ModelPricing.model_key == request.model_key).first()

    if pricing:
        pricing.input_price_per_million = request.input_price_per_million
        pricing.output_price_per_million = request.output_price_per_million
    else:
        pricing = ModelPricing(
            model_key=request.model_key,
            input_price_per_million=request.input_price_per_million,
            output_price_per_million=request.output_price_per_million,
        )
        db.add(pricing)

    db.commit()
    db.refresh(pricing)

    return PricingResponse(
        model_key=pricing.model_key,
        input_price_per_million=pricing.input_price_per_million,
        output_price_per_million=pricing.output_price_per_million,
        created_at=pricing.created_at.isoformat(),
        updated_at=pricing.updated_at.isoformat(),
    )


@router.get("")
async def list_pricing(
    db: Annotated[Session, Depends(get_db)],
    skip: int = 0,
    limit: int = 100,
) -> list[PricingResponse]:
    """List all model pricing."""
    pricings = db.query(ModelPricing).offset(skip).limit(limit).all()

    return [
        PricingResponse(
            model_key=pricing.model_key,
            input_price_per_million=pricing.input_price_per_million,
            output_price_per_million=pricing.output_price_per_million,
            created_at=pricing.created_at.isoformat(),
            updated_at=pricing.updated_at.isoformat(),
        )
        for pricing in pricings
    ]


@router.get("/{model_key}")
async def get_pricing(
    model_key: str,
    db: Annotated[Session, Depends(get_db)],
) -> PricingResponse:
    """Get pricing for a specific model."""
    pricing = db.query(ModelPricing).filter(ModelPricing.model_key == model_key).first()

    if not pricing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pricing for model '{model_key}' not found",
        )

    return PricingResponse(
        model_key=pricing.model_key,
        input_price_per_million=pricing.input_price_per_million,
        output_price_per_million=pricing.output_price_per_million,
        created_at=pricing.created_at.isoformat(),
        updated_at=pricing.updated_at.isoformat(),
    )


@router.delete("/{model_key}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(verify_master_key)])
async def delete_pricing(
    model_key: str,
    db: Annotated[Session, Depends(get_db)],
) -> None:
    """Delete pricing for a model."""
    pricing = db.query(ModelPricing).filter(ModelPricing.model_key == model_key).first()

    if not pricing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pricing for model '{model_key}' not found",
        )

    db.delete(pricing)
    db.commit()
