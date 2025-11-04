"""Tests for pricing configuration from config file."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from any_llm.gateway.config import GatewayConfig, PricingConfig
from any_llm.gateway.db import ModelPricing, get_db
from any_llm.gateway.server import create_app


def test_pricing_loaded_from_config(postgres_url: str, test_db: Session) -> None:
    """Test that pricing is loaded from config file on startup."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        providers={"openai": {"api_key": "test-key"}},
        pricing={
            "openai:gpt-4": PricingConfig(
                input_price_per_million=30.0,
                output_price_per_million=60.0,
            ),
            "openai:gpt-3.5-turbo": PricingConfig(
                input_price_per_million=0.5,
                output_price_per_million=1.5,
            ),
        },
    )

    app = create_app(config)

    def override_get_db() -> Any:
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app):
        # Check GPT-4 pricing
        pricing = test_db.query(ModelPricing).filter(ModelPricing.model_key == "openai:gpt-4").first()
        assert pricing is not None, "GPT-4 pricing should be loaded from config"
        assert pricing.input_price_per_million == 30.0
        assert pricing.output_price_per_million == 60.0

        # Check GPT-3.5 pricing
        pricing = test_db.query(ModelPricing).filter(ModelPricing.model_key == "openai:gpt-3.5-turbo").first()
        assert pricing is not None, "GPT-3.5-turbo pricing should be loaded from config"
        assert pricing.input_price_per_million == 0.5
        assert pricing.output_price_per_million == 1.5


def test_database_pricing_takes_precedence(postgres_url: str, test_db: Session) -> None:
    """Test that existing database pricing is not overwritten by config."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        providers={"openai": {"api_key": "test-key"}},
        pricing={
            "openai:gpt-4": PricingConfig(
                input_price_per_million=30.0,
                output_price_per_million=60.0,
            ),
        },
    )

    # Pre-populate database with different pricing
    existing_pricing = ModelPricing(
        model_key="openai:gpt-4",
        input_price_per_million=25.0,
        output_price_per_million=50.0,
    )
    test_db.add(existing_pricing)
    test_db.commit()

    # Create app (which loads config pricing)
    app = create_app(config)

    def override_get_db() -> Any:
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app):
        # Check that database pricing was preserved
        pricing = test_db.query(ModelPricing).filter(ModelPricing.model_key == "openai:gpt-4").first()
        assert pricing is not None
        # Should keep database values, not config values
        assert pricing.input_price_per_million == 25.0
        assert pricing.output_price_per_million == 50.0


def test_pricing_validation_requires_configured_provider(postgres_url: str, test_db: Session) -> None:
    """Test that pricing initialization fails if provider is not configured."""
    # Config with pricing for a provider that's not in providers list
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        providers={"openai": {"api_key": "test-key"}},
        pricing={
            "anthropic:claude-3-opus": PricingConfig(
                input_price_per_million=15.0,
                output_price_per_million=75.0,
            ),
        },
    )

    # Should raise ValueError when trying to initialize pricing
    with pytest.raises(ValueError, match="provider 'anthropic' is not configured"):
        create_app(config)


def test_pricing_initialization_with_no_config(postgres_url: str, test_db: Session) -> None:
    """Test that app starts successfully when no pricing is configured."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        providers={"openai": {"api_key": "test-key"}},
        pricing={},  # Empty pricing
    )

    app = create_app(config)

    def override_get_db() -> Any:
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app):
        # App should start successfully
        # No pricing should be in database
        pricing_count = test_db.query(ModelPricing).count()
        assert pricing_count == 0, "No pricing should be loaded when config is empty"
