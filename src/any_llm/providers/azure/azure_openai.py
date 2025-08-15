import os
from openai import AzureOpenAI

from any_llm.providers.openai.base import BaseOpenAIProvider

class AzureOpenAIProvider(BaseOpenAIProvider):
    """Azure OpenAI Provider."""

    ENV_API_KEY_NAME = "AZURE_OPENAI_API_KEY"
    PROVIDER_NAME = "azure_openai"
    PROVIDER_DOCUMENTATION_URL = "https://learn.microsoft.com/en-us/azure/ai-foundry/openai/overview"
    SUPPORTS_RESPONSES = True

    def _get_client(self) -> AzureOpenAI:
        return AzureOpenAI(
            base_url=self.config.api_base or self.API_BASE or os.getenv("OPENAI_API_BASE"),
            api_key=self.config.api_key,
        )
