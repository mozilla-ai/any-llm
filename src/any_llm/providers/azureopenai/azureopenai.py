import os
from typing import Any

from openai import AsyncAzureOpenAI

from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.openai.base import BaseOpenAIProvider


class AzureopenaiProvider(BaseOpenAIProvider):
    """Azure OpenAI AnyLLM."""

    ENV_API_KEY_NAME = "AZURE_OPENAI_API_KEY"
    PROVIDER_NAME = "azureopenai"
    PROVIDER_DOCUMENTATION_URL = "https://learn.microsoft.com/en-us/azure/ai-foundry/"
    SUPPORTS_RESPONSES = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_COMPLETION_PDF = False

    DEFAULT_API_VERSION = "preview"

    client: AsyncAzureOpenAI

    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        return api_key or os.getenv(self.ENV_API_KEY_NAME)

    def _init_client(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        api_version = kwargs.pop("api_version", None) or os.getenv("OPENAI_API_VERSION", self.DEFAULT_API_VERSION)
        azure_ad_token = kwargs.pop("azure_ad_token", None) or os.getenv("AZURE_OPENAI_AD_TOKEN")

        azure_endpoint = api_base or kwargs.pop("azure_endpoint", None) or os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise MissingApiKeyError(self.PROVIDER_NAME, "AZURE_OPENAI_ENDPOINT")

        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            azure_ad_token=azure_ad_token,
            **kwargs,
        )
