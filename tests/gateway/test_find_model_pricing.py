"""Tests for the shared find_model_pricing helper."""

from sqlalchemy.orm import Session

from any_llm.gateway.db import ModelPricing
from any_llm.gateway.pricing import find_model_pricing


def test_find_pricing_colon_format(test_db: Session) -> None:
    """Test lookup with canonical colon-separated key."""
    test_db.add(ModelPricing(model_key="openai:gpt-4", input_price_per_million=30.0, output_price_per_million=60.0))
    test_db.commit()

    pricing = find_model_pricing(test_db, "openai", "gpt-4")
    assert pricing is not None
    assert pricing.input_price_per_million == 30.0


def test_find_pricing_legacy_slash_fallback(test_db: Session) -> None:
    """Test fallback to legacy slash-separated key when colon key is missing."""
    test_db.add(ModelPricing(model_key="openai/gpt-4", input_price_per_million=30.0, output_price_per_million=60.0))
    test_db.commit()

    pricing = find_model_pricing(test_db, "openai", "gpt-4")
    assert pricing is not None
    assert pricing.model_key == "openai/gpt-4"


def test_find_pricing_no_provider(test_db: Session) -> None:
    """Test lookup without a provider uses model name directly."""
    test_db.add(ModelPricing(model_key="gpt-4", input_price_per_million=30.0, output_price_per_million=60.0))
    test_db.commit()

    pricing = find_model_pricing(test_db, None, "gpt-4")
    assert pricing is not None
    assert pricing.model_key == "gpt-4"


def test_find_pricing_not_found(test_db: Session) -> None:
    """Test that None is returned when no pricing exists."""
    pricing = find_model_pricing(test_db, "openai", "nonexistent-model")
    assert pricing is None


def test_find_pricing_colon_preferred_over_slash(test_db: Session) -> None:
    """Test that colon format is returned when both formats exist."""
    test_db.add(ModelPricing(model_key="openai:gpt-4", input_price_per_million=30.0, output_price_per_million=60.0))
    test_db.add(ModelPricing(model_key="openai/gpt-4", input_price_per_million=10.0, output_price_per_million=20.0))
    test_db.commit()

    pricing = find_model_pricing(test_db, "openai", "gpt-4")
    assert pricing is not None
    assert pricing.model_key == "openai:gpt-4"
    assert pricing.input_price_per_million == 30.0
