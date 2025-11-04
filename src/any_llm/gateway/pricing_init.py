"""Pricing initialization from configuration."""

from sqlalchemy.orm import Session

from any_llm.any_llm import AnyLLM
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import ModelPricing
from any_llm.gateway.log_config import logger


def initialize_pricing_from_config(config: GatewayConfig, db: Session) -> None:
    """Initialize model pricing from configuration file.

    Loads pricing from config.pricing and stores it in the database.
    Database pricing takes precedence - if pricing exists in DB, it is not overwritten.

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

    for model_key, pricing_config in config.pricing.items():
        provider, _ = AnyLLM.split_model_provider(model_key)

        if provider.value not in config.providers:
            msg = (
                f"Cannot set pricing for model '{model_key}': "
                f"provider '{provider}' is not configured in the providers section"
            )
            raise ValueError(msg)

        input_price = pricing_config.input_price_per_million
        output_price = pricing_config.output_price_per_million

        existing_pricing = db.query(ModelPricing).filter(ModelPricing.model_key == model_key).first()

        if existing_pricing:
            logger.warning(
                f"Pricing for model '{model_key}' already exists in database. "
                f"Keeping database value (input: ${existing_pricing.input_price_per_million}/M, "
                f"output: ${existing_pricing.output_price_per_million}/M). "
                f"To update, use the pricing API or delete the existing entry."
            )
            continue

        new_pricing = ModelPricing(
            model_key=model_key,
            input_price_per_million=input_price,
            output_price_per_million=output_price,
        )
        db.add(new_pricing)
        logger.info(f"Added pricing for '{model_key}': input=${input_price}/M, output=${output_price}/M")

    db.commit()
    logger.info("Pricing initialization complete")
