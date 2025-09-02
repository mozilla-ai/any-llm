"""Azure OpenAI Provider."""

import os

from openai import AsyncOpenAI, OpenAI

from any_llm.providers.openai.base import BaseOpenAIProvider


class AzureopenaiProvider(BaseOpenAIProvider):
    """Azure OpenAI Provider."""

    ENV_API_KEY_NAME = "AZURE_OPENAI_API_KEY"
    PROVIDER_NAME = "azureopenai"
    PROVIDER_DOCUMENTATION_URL = "https://learn.microsoft.com/en-us/azure/ai-foundry/"
    SUPPORTS_RESPONSES = True
    SUPPORTS_LIST_MODELS = True

    def _get_client(self, sync: bool = False) -> AsyncOpenAI | OpenAI:
        _client_class = OpenAI if sync else AsyncOpenAI
        return _client_class(
            base_url=self.config.api_base or self.API_BASE or os.getenv("AZURE_OPENAI_API_BASE"),
            api_key=self.config.api_key,
            **(self.config.client_args if self.config.client_args else {}),
            default_query={"api-version": "preview"},
        )
