import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

API_KEY_HEADER = "X-AnyLLM-Key"


class PricingConfig(BaseModel):
    """Model pricing configuration."""

    input_price_per_million: float
    output_price_per_million: float


class GatewayConfig(BaseSettings):
    """Gateway configuration with support for YAML files and environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/any_llm_gateway",
        description="Database connection URL for PostgreSQL",
    )
    auto_migrate: bool = Field(
        default=True,
        description="Automatically run database migrations on startup",
    )
    host: str = Field(default="0.0.0.0", description="Host to bind the server to")  # noqa: S104
    port: int = Field(default=8000, description="Port to bind the server to")
    master_key: str | None = Field(default=None, description="Master key for protecting management endpoints")
    providers: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Pre-configured provider credentials"
    )
    pricing: dict[str, PricingConfig] = Field(
        default_factory=dict,
        description="Pre-configured model USD pricing (model_key -> {input_price_per_million, output_price_per_million})",
    )


def load_config(config_path: str | None = None) -> GatewayConfig:
    """Load configuration from file and environment variables.

    Args:
        config_path: Optional path to YAML config file

    Returns:
        GatewayConfig instance with merged configuration

    """
    config_dict: dict[str, Any] = {}

    if config_path and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config_dict = _resolve_env_vars(yaml_config)

    return GatewayConfig(**config_dict)


def _resolve_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve environment variable references in config.

    Supports ${VAR_NAME} syntax in string values.
    """
    if isinstance(config, dict):
        return {key: _resolve_env_vars(value) for key, value in config.items()}
    if isinstance(config, list):
        return [_resolve_env_vars(item) for item in config]
    if isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        env_var = config[2:-1]
        return os.getenv(env_var, config)
    return config
