"""OpenAI-compatible models listing endpoint."""

import calendar
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from any_llm.gateway.api.deps import get_db, verify_api_key_or_master_key
from any_llm.gateway.models.entities import ModelPricing

router = APIRouter(prefix="/v1", tags=["models"])


class ModelObject(BaseModel):
    """OpenAI-compatible model object."""

    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    """OpenAI-compatible model list response."""

    object: str = "list"
    data: list[ModelObject]


def _model_from_pricing(pricing: ModelPricing) -> ModelObject:
    """Convert a ModelPricing row to an OpenAI-compatible ModelObject."""
    parts = pricing.model_key.split(":", 1)
    owned_by = parts[0] if len(parts) > 1 else "unknown"
    created = int(calendar.timegm(pricing.created_at.utctimetuple()))
    return ModelObject(
        id=pricing.model_key,
        created=created,
        owned_by=owned_by,
    )


@router.get("/models", dependencies=[Depends(verify_api_key_or_master_key)])
async def list_models(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelListResponse:
    """List all available models.

    Returns models derived from the model_pricing table in an
    OpenAI-compatible format.
    """
    pricings = (await db.execute(select(ModelPricing).order_by(ModelPricing.model_key))).scalars().all()
    return ModelListResponse(data=[_model_from_pricing(p) for p in pricings])


@router.get("/models/{model_id:path}", dependencies=[Depends(verify_api_key_or_master_key)])
async def get_model(
    model_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelObject:
    """Get details for a specific model."""
    pricing = (await db.execute(select(ModelPricing).where(ModelPricing.model_key == model_id))).scalar_one_or_none()

    if not pricing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found",
        )

    return _model_from_pricing(pricing)
