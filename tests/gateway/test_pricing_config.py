"""Tests for pricing configuration from config file."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from any_llm.gateway.config import GatewayConfig, PricingConfig
from any_llm.gateway.db import ModelPricing, get_db
from any_llm.gateway.db.models import UsageLog
from any_llm.gateway.routes.chat import _log_usage
from any_llm.gateway.server import create_app
from any_llm.types.completion import CompletionUsage


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


def test_pricing_loaded_from_config_normalizes_legacy_slash_format(postgres_url: str, test_db: Session) -> None:
    """Test that pricing configured with legacy slash format is normalized to colon format."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        providers={"openai": {"api_key": "test-key"}},
        pricing={
            "openai/gpt-4": PricingConfig(
                input_price_per_million=30.0,
                output_price_per_million=60.0,
            ),
        },
    )

    app = create_app(config)

    def override_get_db() -> Any:
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app):
        # Pricing should be stored with canonical colon format, not slash
        pricing_slash = test_db.query(ModelPricing).filter(ModelPricing.model_key == "openai/gpt-4").first()
        assert pricing_slash is None, "Pricing should not be stored with legacy slash format"

        pricing_colon = test_db.query(ModelPricing).filter(ModelPricing.model_key == "openai:gpt-4").first()
        assert pricing_colon is not None, "Pricing should be stored with canonical colon format"
        assert pricing_colon.input_price_per_million == 30.0
        assert pricing_colon.output_price_per_million == 60.0


def test_set_pricing_api_normalizes_legacy_slash_format(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Test that the pricing API normalizes legacy slash format to colon format."""
    response = client.post(
        "/v1/pricing",
        json={
            "model_key": "gemini/gemini-2.5-flash",
            "input_price_per_million": 0.075,
            "output_price_per_million": 0.30,
        },
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["model_key"] == "gemini:gemini-2.5-flash", "API should normalize slash to colon format"


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


@pytest.mark.asyncio
async def test_log_usage_finds_pricing_with_legacy_slash_format(test_db: Session) -> None:
    """Test that _log_usage falls back to legacy slash format when colon format is not found."""
    # Simulate pricing stored with legacy slash format (e.g., from before normalization fix)
    legacy_pricing = ModelPricing(
        model_key="openai/gpt-4",
        input_price_per_million=30.0,
        output_price_per_million=60.0,
    )
    test_db.add(legacy_pricing)
    test_db.commit()

    usage = CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    await _log_usage(
        db=test_db,
        api_key_obj=None,
        model="gpt-4",
        provider="openai",
        endpoint="/v1/chat/completions",
        usage_override=usage,
    )

    log = test_db.query(UsageLog).first()
    assert log is not None
    assert log.cost is not None, "Cost should be calculated via legacy slash format fallback"
    expected_cost = (1000 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0
    assert abs(log.cost - expected_cost) < 0.0001


@pytest.mark.asyncio
async def test_log_usage_finds_pricing_with_colon_format(test_db: Session) -> None:
    """Test that _log_usage finds pricing with canonical colon format."""
    pricing = ModelPricing(
        model_key="openai:gpt-4",
        input_price_per_million=30.0,
        output_price_per_million=60.0,
    )
    test_db.add(pricing)
    test_db.commit()

    usage = CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    await _log_usage(
        db=test_db,
        api_key_obj=None,
        model="gpt-4",
        provider="openai",
        endpoint="/v1/chat/completions",
        usage_override=usage,
    )

    log = test_db.query(UsageLog).first()
    assert log is not None
    assert log.cost is not None, "Cost should be calculated with canonical colon format"
    expected_cost = (1000 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0
    assert abs(log.cost - expected_cost) < 0.0001
