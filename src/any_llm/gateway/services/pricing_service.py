"""Shared pricing lookup utilities."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from any_llm.gateway.models.entities import ModelPricing


async def find_model_pricing(db: AsyncSession, provider: str | None, model: str) -> ModelPricing | None:
    """Look up model pricing, falling back to legacy slash-separated key format.

    Args:
        db: Database session
        provider: Provider name (e.g., "openai") or None
        model: Model name (e.g., "gpt-4")

    Returns:
        ModelPricing object if found, None otherwise

    """
    model_key = f"{provider}:{model}" if provider else model
    pricing = (await db.execute(select(ModelPricing).where(ModelPricing.model_key == model_key))).scalar_one_or_none()
    if not pricing and provider:
        legacy_key = f"{provider}/{model}"
        pricing = (
            await db.execute(select(ModelPricing).where(ModelPricing.model_key == legacy_key))
        ).scalar_one_or_none()
    return pricing
