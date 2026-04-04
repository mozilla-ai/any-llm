"""Repository helpers for model pricing lookups."""

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from any_llm.gateway.db import ModelPricing


def get_model_pricing(
    db: Session,
    provider: str | None,
    model: str,
    as_of: datetime | None = None,
) -> ModelPricing | None:
    """Look up the effective model pricing at a given point in time.

    Selects the most recent price entry where effective_at <= as_of.
    Falls back to legacy slash-separated key format if no match is found.

    Args:
        db: Database session
        provider: Provider name (e.g., "openai") or None
        model: Model name (e.g., "gpt-4")
        as_of: Timestamp to evaluate pricing at. Defaults to now.

    Returns:
        ModelPricing object if found, None otherwise

    """
    if as_of is None:
        as_of = datetime.now(UTC)

    model_key = f"{provider}:{model}" if provider else model

    pricing = _find_effective_pricing(db, model_key, as_of)
    if not pricing and provider:
        legacy_key = f"{provider}/{model}"
        pricing = _find_effective_pricing(db, legacy_key, as_of)
    return pricing


def _find_effective_pricing(db: Session, model_key: str, as_of: datetime) -> ModelPricing | None:
    """Find the price entry in effect at a given timestamp for a model key."""
    return (
        db.query(ModelPricing)
        .filter(ModelPricing.model_key == model_key, ModelPricing.effective_at <= as_of)
        .order_by(ModelPricing.effective_at.desc())
        .first()
    )
