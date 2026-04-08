"""Pricing initialization from configuration."""

from datetime import UTC, datetime

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from any_llm.any_llm import AnyLLM
from any_llm.gateway.core.config import GatewayConfig
from any_llm.gateway.log_config import logger
from any_llm.gateway.models.entities import ModelPricing


def initialize_pricing_from_config(config: GatewayConfig, db: Session) -> None:
    """Initialize model pricing from configuration file.

    Loads pricing from config.pricing and stores it in the database.
    Each entry is keyed by (model_key, effective_at). If an entry with the
    same composite key already exists in the database, it is not overwritten.

    Args:
        config: Gateway configuration containing pricing definitions
        db: Database session

    Raises:
        ValueError: If pricing is defined for a model from an unconfigured provider

    """
    if not config.pricing:
        logger.debug("No pricing configuration found in config file")
        return

    logger.info(f"Loading pricing configuration for {len(config.pricing)} model(s)")

    for raw_model_key, pricing_config in config.pricing.items():
        provider, model_name = AnyLLM.split_model_provider(raw_model_key)
        model_key = f"{provider.value}:{model_name}"

        if provider.value not in config.providers:
            msg = (
                f"Cannot set pricing for model '{model_key}': "
                f"provider '{provider}' is not configured in the providers section"
            )
            raise ValueError(msg)

        effective_at = pricing_config.effective_at or datetime.now(UTC)
        input_price = pricing_config.input_price_per_million
        output_price = pricing_config.output_price_per_million

        existing = (
            db.query(ModelPricing)
            .filter(
                ModelPricing.model_key == model_key,
                ModelPricing.effective_at == effective_at,
            )
            .first()
        )

        if existing:
            logger.debug(
                f"Pricing for '{model_key}' effective {effective_at.isoformat()} already exists in database, skipping"
            )
            continue

        new_pricing = ModelPricing(
            model_key=model_key,
            effective_at=effective_at,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
        )
        db.add(new_pricing)
        logger.info(
            f"Added pricing for '{model_key}' effective {effective_at.isoformat()}: "
            f"input=${input_price}/M, output=${output_price}/M"
        )

    try:
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise
    logger.info("Pricing initialization complete")
