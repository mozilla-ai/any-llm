import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

API_KEY_HEADER = "X-AnyLLM-Key"


class PricingConfig(BaseModel):
    """Model pricing configuration."""

    input_price_per_million: float = Field(ge=0)
    output_price_per_million: float = Field(ge=0)


class GatewayConfig(BaseSettings):
    """Gateway configuration with support for YAML files and environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = Field(
        default="sqlite:///./any-llm-gateway.db",
        description="Database connection URL (SQLite default for local use; PostgreSQL recommended for production)",
    )
    auto_migrate: bool = Field(
        default=True,
        description="Automatically run database migrations on startup",
    )
    db_pool_size: int = Field(
        default=10,
        ge=1,
        description=(
            "Number of persistent connections maintained in the DB pool per worker. "
            "SQLAlchemy default is 5. For async workloads with many concurrent in-flight "
            "queries, 10-20 is typical."
        ),
    )
    db_max_overflow: int = Field(
        default=20,
        ge=0,
        description=(
            "Extra connections the pool can open above db_pool_size during bursts. "
            "Total max connections per worker = db_pool_size + db_max_overflow. "
            "SQLAlchemy default is 10."
        ),
    )
    db_pool_timeout: float = Field(
        default=30.0,
        ge=0,
        description="Seconds to wait for an available connection before raising TimeoutError.",
    )
    db_pool_recycle: int = Field(
        default=-1,
        description=(
            "Recycle connections older than this many seconds. -1 disables recycling. "
            "Set to 300-600 behind proxies/load-balancers that drop idle TCP connections."
        ),
    )
    host: str = Field(default="0.0.0.0", description="Host to bind the server to")  # noqa: S104
    port: int = Field(default=8000, description="Port to bind the server to")
    master_key: str | None = Field(default=None, description="Master key for protecting management endpoints")
    rate_limit_rpm: int | None = Field(
        default=None, ge=1, description="Maximum requests per minute per user (None disables rate limiting)"
    )
    cors_allow_origins: list[str] = Field(
        default_factory=list, description="Allowed CORS origins (empty list disables CORS)"
    )
    providers: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Pre-configured provider credentials"
    )
    pricing: dict[str, PricingConfig] = Field(
        default_factory=dict,
        description="Pre-configured model USD pricing (model_key -> {input_price_per_million, output_price_per_million})",
    )
    enable_metrics: bool = Field(
        default=False,
        description="Enable Prometheus metrics endpoint at /metrics",
    )
    bootstrap_api_key: bool = Field(
        default=True,
        description="Create a first-use API key on startup when no API keys exist",
    )
    log_writer_strategy: str = Field(
        default="single",
        description=(
            "How usage log rows are written to the DB. "
            "'single' (default): write each event inline on the request path, one transaction per event. "
            "'batch': enqueue events and flush in batches of up to 100 rows or every 1s, whichever "
            "comes first. Write happens in a background task, so the request hot path doesn't "
            "block on log-writing DB round-trips. 'At most once' semantics — a crashed worker "
            "loses up to one batch of in-flight rows."
        ),
    )
    budget_strategy: str = Field(
        default="for_update",
        description=(
            "How per-user budget validation runs on the request hot path. "
            "'for_update' (default): FOR UPDATE held across the entire request, including the LLM "
            "call. This is the historical behavior — kept as the default for backwards compatibility. "
            "'cas' (recommended): lock-free — reset via atomic conditional UPDATE "
            "(WHERE next_budget_reset_at < now). No explicit FOR UPDATE. Hot-path reads are "
            "unlocked, so concurrent requests for the same user never serialize. "
            "'disabled': skip validate_user_budget entirely — no user existence check, no blocked "
            "check, no budget check. Cost tracking via log_usage still runs."
        ),
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

    Raises:
        ValueError: If an environment variable reference cannot be resolved

    """
    if isinstance(config, dict):
        return {key: _resolve_env_vars(value) for key, value in config.items()}
    if isinstance(config, list):
        return [_resolve_env_vars(item) for item in config]
    if isinstance(config, str) and "${" in config:

        def _replace(match: re.Match[str]) -> str:
            env_var = match.group(1)
            value = os.getenv(env_var)
            if value is None:
                msg = f"Environment variable '{env_var}' is not set (referenced in config as '${{{env_var}}}')"
                raise ValueError(msg)
            return value

        return re.sub(r"\$\{([^}]+)}", _replace, config)
    return config
