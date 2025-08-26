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

    @property
    def openai_client(self) -> OpenAI:
        return OpenAI(
            base_url=self.config.api_base or self.API_BASE or os.getenv("AZURE_OPENAI_API_BASE"),
            api_key=self.config.api_key,
            default_query={"api-version": "preview"},
        )

    @property
    def async_openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.config.api_base or self.API_BASE or os.getenv("AZURE_OPENAI_API_BASE"),
            api_key=self.config.api_key,
            default_query={"api-version": "preview"},
        )
