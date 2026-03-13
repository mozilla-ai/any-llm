from __future__ import annotations

import os
from typing import Any

from typing_extensions import override

from any_llm.providers.anthropic.base import BaseAnthropicProvider

MISSING_PACKAGES_ERROR = None
try:
    from anthropic import AsyncAnthropicFoundry
except ImportError as e:
    MISSING_PACKAGES_ERROR = e


class AzureanthropicProvider(BaseAnthropicProvider):
    """
    Azure Anthropic Provider for Claude models through Microsoft Foundry.

    Uses Anthropic's `AsyncAnthropicFoundry` client to access Claude models
    deployed on Azure via Microsoft Foundry.

    Authentication requires an API key (via `AZURE_ANTHROPIC_API_KEY` or the `api_key`
    constructor argument). Optionally accepts a `resource` name (via `AZURE_ANTHROPIC_RESOURCE`
    or the `resource` constructor argument) identifying the Azure Foundry resource.
    """

    PROVIDER_NAME = "azureanthropic"
    ENV_API_KEY_NAME = "AZURE_ANTHROPIC_API_KEY"
    ENV_API_BASE_NAME = "AZURE_ANTHROPIC_API_BASE"
    PROVIDER_DOCUMENTATION_URL = "https://learn.microsoft.com/en-us/azure/ai-foundry/model-inference/concepts/models"

    SUPPORTS_LIST_MODELS = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncAnthropicFoundry

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        resource = kwargs.pop("resource", None) or os.getenv("AZURE_ANTHROPIC_RESOURCE")

        if api_base is not None:
            self.client = AsyncAnthropicFoundry(
                base_url=api_base,
                api_key=api_key,
                **kwargs,
            )
        else:
            self.client = AsyncAnthropicFoundry(
                resource=resource,
                api_key=api_key,
                **kwargs,
            )
