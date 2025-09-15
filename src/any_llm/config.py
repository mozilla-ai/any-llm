from typing import Any

from pydantic import BaseModel


class ClientConfig(BaseModel):
    """Configuration for the underlying client used by the provider."""

    api_key: str | None = None
    api_base: str | None = None
    client_args: dict[str, Any] | None = None
