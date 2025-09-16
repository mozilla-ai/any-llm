from typing import Any

from pydantic import BaseModel, ConfigDict


class ClientConfig(BaseModel):
    """Configuration for the underlying client used by the provider."""

    model_config = ConfigDict(extra="forbid")

    api_key: str | None = None
    api_base: str | None = None
    client_args: dict[str, Any] | None = None
