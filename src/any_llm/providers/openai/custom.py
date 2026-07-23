from __future__ import annotations

import os
from typing import Any

from typing_extensions import override

from any_llm.providers.openai.base import BaseOpenAIProvider


class OpenAICompatibleProvider(BaseOpenAIProvider):
    """Point any-llm at an arbitrary OpenAI-compatible endpoint under its own name.

    Unlike the built-in providers, this one is configured entirely at construction
    time: the caller supplies the endpoint's base URL, instead of relying on a
    registered provider entry. It is the supported path for any OpenAI-compatible
    gateway that any-llm does not ship a dedicated provider for, so nobody is blocked
    from using their endpoint.

    Capability flags follow the OpenAI-compatible defaults from ``BaseOpenAIProvider``;
    a call to an endpoint that does not implement a given capability surfaces the
    provider's own error.

    Prefer ``AnyLLM.create_openai_compatible(...)``, which reports the caller's chosen
    name as the provider identity (via a per-name subclass) rather than masquerading as
    ``openai``. Constructing this class directly reports the generic ``openai_compatible``
    identity.
    """

    PROVIDER_NAME = "openai_compatible"
    PROVIDER_DOCUMENTATION_URL = "https://platform.openai.com/docs/api-reference"
    ENV_API_KEY_NAME = "OPENAI_COMPATIBLE_API_KEY"
    ENV_API_BASE_NAME = "OPENAI_COMPATIBLE_API_BASE"

    def __init__(self, api_base: str, api_key: str | None = None, **kwargs: Any) -> None:
        if not api_base:
            msg = "OpenAICompatibleProvider requires an explicit api_base pointing at the endpoint."
            raise ValueError(msg)
        # Bind the endpoint per instance so the client targets the caller's URL rather
        # than the (unset) class default.
        self.API_BASE = api_base
        super().__init__(api_key=api_key, api_base=api_base, **kwargs)

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        # Custom endpoints may be keyless (local servers) or keyed (hosted gateways).
        # Fall back to the env var when set, then to a placeholder so the OpenAI client
        # accepts the value; never raise, so nobody is blocked from using their endpoint.
        return api_key or os.getenv(self.ENV_API_KEY_NAME) or "no-key-required"
