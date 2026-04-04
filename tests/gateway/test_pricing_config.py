"""Tests for pricing configuration from config file."""

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from any_llm.gateway.api.routes.chat import log_usage
from any_llm.gateway.core.config import GatewayConfig, PricingConfig
from any_llm.gateway.db import ModelPricing, get_db
from any_llm.gateway.main import create_app
from any_llm.gateway.models.entities import UsageLog
from any_llm.gateway.services.pricing_service import find_model_pricing
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
        pricing = test_db.query(ModelPricing).filter(ModelPricing.model_key == "openai:gpt-4").first()
        assert pricing is not None, "GPT-4 pricing should be loaded from config"
        assert pricing.input_price_per_million == 30.0
        assert pricing.output_price_per_million == 60.0
        assert pricing.effective_at is not None

        pricing = test_db.query(ModelPricing).filter(ModelPricing.model_key == "openai:gpt-3.5-turbo").first()
        assert pricing is not None, "GPT-3.5-turbo pricing should be loaded from config"
        assert pricing.input_price_per_million == 0.5
        assert pricing.output_price_per_million == 1.5


def test_pricing_loaded_with_explicit_effective_at(postgres_url: str, test_db: Session) -> None:
    """Test that pricing with explicit effective_at is stored correctly."""
    effective = datetime(2025, 6, 1, tzinfo=UTC)
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
                effective_at=effective,
            ),
        },
    )

    app = create_app(config)

    def override_get_db() -> Any:
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app):
        pricing = test_db.query(ModelPricing).filter(ModelPricing.model_key == "openai:gpt-4").first()
        assert pricing is not None
        assert pricing.effective_at == effective
        assert pricing.input_price_per_million == 30.0


def test_database_pricing_takes_precedence(postgres_url: str, test_db: Session) -> None:
    """Test that existing database pricing with same effective_at is not overwritten."""
    effective = datetime(2025, 1, 1, tzinfo=UTC)
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
                effective_at=effective,
            ),
        },
    )

    existing_pricing = ModelPricing(
        model_key="openai:gpt-4",
        effective_at=effective,
        input_price_per_million=25.0,
        output_price_per_million=50.0,
    )
    test_db.add(existing_pricing)
    test_db.commit()

    app = create_app(config)

    def override_get_db() -> Any:
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app):
        pricing = (
            test_db.query(ModelPricing)
            .filter(ModelPricing.model_key == "openai:gpt-4", ModelPricing.effective_at == effective)
            .first()
        )
        assert pricing is not None
        assert pricing.input_price_per_million == 25.0
        assert pricing.output_price_per_million == 50.0


def test_pricing_validation_requires_configured_provider(postgres_url: str, test_db: Session) -> None:
    """Test that pricing initialization fails if provider is not configured."""
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
    assert "effective_at" in data


def test_set_pricing_api_with_effective_at(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Test that the pricing API accepts an explicit effective_at."""
    effective = "2025-06-01T00:00:00+00:00"
    response = client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4",
            "input_price_per_million": 30.0,
            "output_price_per_million": 60.0,
            "effective_at": effective,
        },
        headers=master_key_header,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["effective_at"] == effective


def test_pricing_history_endpoint(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Test the pricing history endpoint returns all entries for a model."""
    # Create two price entries at different effective dates
    for price, date in [(20.0, "2025-01-01T00:00:00+00:00"), (30.0, "2025-06-01T00:00:00+00:00")]:
        client.post(
            "/v1/pricing",
            json={
                "model_key": "openai:gpt-4",
                "input_price_per_million": price,
                "output_price_per_million": price * 2,
                "effective_at": date,
            },
            headers=master_key_header,
        )

    response = client.get("/v1/pricing/openai:gpt-4/history")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    # Newest first
    assert data[0]["input_price_per_million"] == 30.0
    assert data[1]["input_price_per_million"] == 20.0


def test_pricing_initialization_with_no_config(postgres_url: str, test_db: Session) -> None:
    """Test that app starts successfully when no pricing is configured."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        providers={"openai": {"api_key": "test-key"}},
        pricing={},
    )

    app = create_app(config)

    def override_get_db() -> Any:
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app):
        pricing_count = test_db.query(ModelPricing).count()
        assert pricing_count == 0, "No pricing should be loaded when config is empty"


@pytest.mark.asyncio
async def test_log_usage_finds_pricing_with_legacy_slash_format(test_db: Session) -> None:
    """Test that log_usage falls back to legacy slash format when colon format is not found."""
    legacy_pricing = ModelPricing(
        model_key="openai/gpt-4",
        input_price_per_million=30.0,
        output_price_per_million=60.0,
    )
    test_db.add(legacy_pricing)
    test_db.commit()

    usage = CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    await log_usage(
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
    """Test that log_usage finds pricing with canonical colon format."""
    pricing = ModelPricing(
        model_key="openai:gpt-4",
        input_price_per_million=30.0,
        output_price_per_million=60.0,
    )
    test_db.add(pricing)
    test_db.commit()

    usage = CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    await log_usage(
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


def test_find_model_pricing_effective_at_lookup(test_db: Session) -> None:
    """Test that find_model_pricing returns the correct price for a given timestamp."""
    old_price = ModelPricing(
        model_key="openai:gpt-4",
        effective_at=datetime(2025, 1, 1, tzinfo=UTC),
        input_price_per_million=20.0,
        output_price_per_million=40.0,
    )
    new_price = ModelPricing(
        model_key="openai:gpt-4",
        effective_at=datetime(2025, 6, 1, tzinfo=UTC),
        input_price_per_million=30.0,
        output_price_per_million=60.0,
    )
    test_db.add_all([old_price, new_price])
    test_db.commit()

    # Before any price exists
    result = find_model_pricing(test_db, "openai", "gpt-4", as_of=datetime(2024, 12, 1, tzinfo=UTC))
    assert result is None

    # During old price period
    result = find_model_pricing(test_db, "openai", "gpt-4", as_of=datetime(2025, 3, 1, tzinfo=UTC))
    assert result is not None
    assert result.input_price_per_million == 20.0

    # After new price takes effect
    result = find_model_pricing(test_db, "openai", "gpt-4", as_of=datetime(2025, 7, 1, tzinfo=UTC))
    assert result is not None
    assert result.input_price_per_million == 30.0

    # Exactly on effective_at boundary
    result = find_model_pricing(test_db, "openai", "gpt-4", as_of=datetime(2025, 6, 1, tzinfo=UTC))
    assert result is not None
    assert result.input_price_per_million == 30.0


def test_find_model_pricing_defaults_to_now(test_db: Session) -> None:
    """Test that find_model_pricing defaults to current time when as_of is not provided."""
    pricing = ModelPricing(
        model_key="openai:gpt-4",
        effective_at=datetime(2025, 1, 1, tzinfo=UTC),
        input_price_per_million=30.0,
        output_price_per_million=60.0,
    )
    test_db.add(pricing)
    test_db.commit()

    result = find_model_pricing(test_db, "openai", "gpt-4")
    assert result is not None
    assert result.input_price_per_million == 30.0


def test_find_model_pricing_ignores_future_prices(test_db: Session) -> None:
    """Test that future prices are not returned for current lookups."""
    now = datetime.now(UTC)
    current_price = ModelPricing(
        model_key="openai:gpt-4",
        effective_at=now - timedelta(days=30),
        input_price_per_million=20.0,
        output_price_per_million=40.0,
    )
    future_price = ModelPricing(
        model_key="openai:gpt-4",
        effective_at=now + timedelta(days=30),
        input_price_per_million=50.0,
        output_price_per_million=100.0,
    )
    test_db.add_all([current_price, future_price])
    test_db.commit()

    result = find_model_pricing(test_db, "openai", "gpt-4", as_of=now)
    assert result is not None
    assert result.input_price_per_million == 20.0, "Should return current price, not future"


def test_delete_pricing_specific_effective_at(
    client: TestClient,
    master_key_header: dict[str, str],
) -> None:
    """Test deleting a specific price entry by effective_at."""
    # Create two entries
    for date in ["2025-01-01T00:00:00+00:00", "2025-06-01T00:00:00+00:00"]:
        client.post(
            "/v1/pricing",
            json={
                "model_key": "openai:gpt-4",
                "input_price_per_million": 30.0,
                "output_price_per_million": 60.0,
                "effective_at": date,
            },
            headers=master_key_header,
        )

    # Delete only the old one
    response = client.delete(
        "/v1/pricing/openai:gpt-4",
        params={"effective_at": "2025-01-01T00:00:00+00:00"},
        headers=master_key_header,
    )
    assert response.status_code == 204

    # History should have only one entry left
    history = client.get("/v1/pricing/openai:gpt-4/history")
    assert len(history.json()) == 1
    assert history.json()[0]["effective_at"] == "2025-06-01T00:00:00+00:00"
