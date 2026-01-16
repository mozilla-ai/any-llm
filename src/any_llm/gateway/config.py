import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

API_KEY_HEADER = "X-AnyLLM-Key"


class PricingConfig(BaseModel):
    """Model pricing configuration."""

    input_price_per_million: float
    output_price_per_million: float


class GatewayConfig(BaseSettings):
    """Gateway configuration with support for YAML files and environment variables.

    All configuration parameters can be set via environment variables with GATEWAY_ prefix:
    - Simple values: GATEWAY_HOST, GATEWAY_PORT, GATEWAY_DATABASE_URL, etc.
    - Boolean values: GATEWAY_AUTO_MIGRATE=true/false
    - Complex structures (JSON): GATEWAY_PROVIDERS='{"openai": {"api_key": "sk-..."}}'
    - Complex structures (JSON): GATEWAY_PRICING='{"openai:gpt-4": {"input_price_per_million": 30, "output_price_per_million": 60}}'

    Environment variables take precedence over YAML config values.
    """

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        case_sensitive=False,
        env_nested_delimiter="__",
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

    @field_validator("providers", mode="before")
    @classmethod
    def parse_providers(cls, v: Any) -> dict[str, dict[str, Any]]:
        """Parse providers from JSON string or return dict as-is."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError as e:
                msg = f"Invalid JSON in GATEWAY_PROVIDERS: {e}"
                raise ValueError(msg) from e
            else:
                if not isinstance(parsed, dict):
                    msg = "GATEWAY_PROVIDERS must be a JSON object"
                    raise ValueError(msg)
                return parsed
        return v if isinstance(v, dict) else {}

    @field_validator("pricing", mode="before")
    @classmethod
    def parse_pricing(cls, v: Any) -> dict[str, dict[str, float]]:
        """Parse pricing from JSON string or return dict as-is."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError as e:
                msg = f"Invalid JSON in GATEWAY_PRICING: {e}"
                raise ValueError(msg) from e
            else:
                if not isinstance(parsed, dict):
                    msg = "GATEWAY_PRICING must be a JSON object"
                    raise ValueError(msg)
                return parsed
        return v if isinstance(v, dict) else {}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings source precedence.

        Order (highest to lowest priority):
        1. Environment variables
        2. Init settings (from YAML config file)
        3. .env file
        4. Secrets directory
        """
        return env_settings, init_settings, dotenv_settings, file_secret_settings


def load_config(config_path: str | None = None) -> GatewayConfig:
    """Load configuration from file and environment variables.

    Environment variables take precedence over YAML config values.
    All config parameters support GATEWAY_ prefixed env vars.

    Args:
        config_path: Optional path to YAML config file

    Returns:
        GatewayConfig instance with merged configuration

    Example:
        # Using environment variables only (no config file needed):
        export GATEWAY_HOST="0.0.0.0"
        export GATEWAY_PORT=8000
        export GATEWAY_DATABASE_URL="postgresql://..."
        export GATEWAY_MASTER_KEY="your-secret-key"
        export GATEWAY_PROVIDERS='{"openai": {"api_key": "sk-..."}}'
        export GATEWAY_PRICING='{"openai:gpt-4": {"input_price_per_million": 30, "output_price_per_million": 60}}'

    """
    config_dict: dict[str, Any] = {}

    # Load from YAML file if provided
    if config_path and Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config:
                config_dict = _resolve_env_vars(yaml_config)

    # GatewayConfig (BaseSettings) will automatically load environment variables
    # and they will take precedence over the config_dict values
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
