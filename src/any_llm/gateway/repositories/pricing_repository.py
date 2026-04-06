"""Repository helpers for model pricing lookups."""

from sqlalchemy.orm import Session

from any_llm.gateway.db import ModelPricing


def get_model_pricing(db: Session, provider: str | None, model: str) -> ModelPricing | None:
    """Look up model pricing, falling back to legacy slash-separated key format.

    Args:
        db: Database session
        provider: Provider name (e.g., "openai") or None
        model: Model name (e.g., "gpt-4")

    Returns:
        ModelPricing object if found, None otherwise

    """
    model_key = f"{provider}:{model}" if provider else model
    pricing = db.query(ModelPricing).filter(ModelPricing.model_key == model_key).first()
    if not pricing and provider:
        legacy_key = f"{provider}/{model}"
        pricing = db.query(ModelPricing).filter(ModelPricing.model_key == legacy_key).first()
    return pricing
