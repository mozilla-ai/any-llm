"""Tests for pricing configuration from config file."""

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from any_llm.gateway.api.routes.chat import log_usage
from any_llm.gateway.core.config import GatewayConfig, PricingConfig
from any_llm.gateway.db import ModelPricing, get_db
from any_llm.gateway.main import create_app
from any_llm.gateway.models.entities import UsageLog
from any_llm.types.completion import CompletionUsage


def _make_override(session: AsyncSession) -> Any:
    async def override_get_db() -> AsyncGenerator[AsyncSession]:
        yield session

    return override_get_db


@pytest.mark.asyncio
async def test_pricing_loaded_from_config(postgres_url: str, test_db: AsyncSession) -> None:
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
    app.dependency_overrides[get_db] = _make_override(test_db)

    with TestClient(app):
        pricing = (
            await test_db.execute(select(ModelPricing).where(ModelPricing.model_key == "openai:gpt-4"))
        ).scalar_one_or_none()
        assert pricing is not None, "GPT-4 pricing should be loaded from config"
        assert pricing.input_price_per_million == 30.0
        assert pricing.output_price_per_million == 60.0

        pricing = (
            await test_db.execute(select(ModelPricing).where(ModelPricing.model_key == "openai:gpt-3.5-turbo"))
        ).scalar_one_or_none()
        assert pricing is not None, "GPT-3.5-turbo pricing should be loaded from config"
        assert pricing.input_price_per_million == 0.5
        assert pricing.output_price_per_million == 1.5


@pytest.mark.asyncio
async def test_database_pricing_takes_precedence(postgres_url: str, test_db: AsyncSession) -> None:
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

    existing_pricing = ModelPricing(
        model_key="openai:gpt-4",
        input_price_per_million=25.0,
        output_price_per_million=50.0,
    )
    test_db.add(existing_pricing)
    await test_db.commit()

    app = create_app(config)
    app.dependency_overrides[get_db] = _make_override(test_db)

    with TestClient(app):
        pricing = (
            await test_db.execute(select(ModelPricing).where(ModelPricing.model_key == "openai:gpt-4"))
        ).scalar_one_or_none()
        assert pricing is not None
        assert pricing.input_price_per_million == 25.0
        assert pricing.output_price_per_million == 50.0


def test_pricing_validation_requires_configured_provider(postgres_url: str) -> None:
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

    # ValueError is raised from the lifespan startup; TestClient triggers it on context entry.
    app = create_app(config)
    with pytest.raises(ValueError, match="provider 'anthropic' is not configured"):
        with TestClient(app):
            pass


@pytest.mark.asyncio
async def test_pricing_loaded_from_config_normalizes_legacy_slash_format(
    postgres_url: str, test_db: AsyncSession
) -> None:
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
    app.dependency_overrides[get_db] = _make_override(test_db)

    with TestClient(app):
        pricing_slash = (
            await test_db.execute(select(ModelPricing).where(ModelPricing.model_key == "openai/gpt-4"))
        ).scalar_one_or_none()
        assert pricing_slash is None, "Pricing should not be stored with legacy slash format"

        pricing_colon = (
            await test_db.execute(select(ModelPricing).where(ModelPricing.model_key == "openai:gpt-4"))
        ).scalar_one_or_none()
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


@pytest.mark.asyncio
async def test_pricing_initialization_with_no_config(postgres_url: str, test_db: AsyncSession) -> None:
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
    app.dependency_overrides[get_db] = _make_override(test_db)

    with TestClient(app):
        pricing_count = (await test_db.execute(select(func.count()).select_from(ModelPricing))).scalar_one()
        assert pricing_count == 0, "No pricing should be loaded when config is empty"


@pytest.mark.asyncio
async def test_log_usage_finds_pricing_with_legacy_slash_format(test_db: AsyncSession) -> None:
    """Test that _log_usage falls back to legacy slash format when colon format is not found."""
    legacy_pricing = ModelPricing(
        model_key="openai/gpt-4",
        input_price_per_million=30.0,
        output_price_per_million=60.0,
    )
    test_db.add(legacy_pricing)
    await test_db.commit()

    usage = CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    await log_usage(
        db=test_db,
        api_key_obj=None,
        model="gpt-4",
        provider="openai",
        endpoint="/v1/chat/completions",
        usage_override=usage,
    )

    log = (await test_db.execute(select(UsageLog))).scalars().first()
    assert log is not None
    assert log.cost is not None, "Cost should be calculated via legacy slash format fallback"
    expected_cost = (1000 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0
    assert abs(log.cost - expected_cost) < 0.0001


@pytest.mark.asyncio
async def test_log_usage_finds_pricing_with_colon_format(test_db: AsyncSession) -> None:
    """Test that _log_usage finds pricing with canonical colon format."""
    pricing = ModelPricing(
        model_key="openai:gpt-4",
        input_price_per_million=30.0,
        output_price_per_million=60.0,
    )
    test_db.add(pricing)
    await test_db.commit()

    usage = CompletionUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)

    await log_usage(
        db=test_db,
        api_key_obj=None,
        model="gpt-4",
        provider="openai",
        endpoint="/v1/chat/completions",
        usage_override=usage,
    )

    log = (await test_db.execute(select(UsageLog))).scalars().first()
    assert log is not None
    assert log.cost is not None, "Cost should be calculated with canonical colon format"
    expected_cost = (1000 / 1_000_000) * 30.0 + (500 / 1_000_000) * 60.0
    assert abs(log.cost - expected_cost) < 0.0001
