import os
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncAzureOpenAI, AsyncStream
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseStreamEvent as OpenAIResponseStreamEvent
from openresponses_types import CreateResponseBody, ResponseResource

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

    async def _aresponses(
        self, params: CreateResponseBody, **kwargs: Any
    ) -> ResponseResource | AsyncIterator[dict[str, Any]]:
        """Call OpenAI Responses API and return OpenResponses types."""
        response: OpenAIResponse | AsyncStream[OpenAIResponseStreamEvent] = await self.client.responses.create(
            **params.model_dump(exclude_none=True), **kwargs
        )

        if isinstance(response, OpenAIResponse):
            response_dict = response.model_dump()
            # OpenResponses requires that nothing be empty but azure is empty presence_penalty and frequency_penalty
            # openai extends their own spec via model_extra to do this, while azure doesn't...
            if response_dict.get("presence_penalty") is None:
                response_dict["presence_penalty"] = 0
            if response_dict.get("frequency_penalty") is None:
                response_dict["frequency_penalty"] = 0
            return ResponseResource.model_validate(response_dict)

        if isinstance(response, AsyncStream):

            async def stream_iterator() -> AsyncIterator[dict[str, Any]]:
                async for event in response:
                    yield event.model_dump()

            return stream_iterator()

        msg = f"Responses API returned an unexpected type: {type(response)}"
        raise ValueError(msg)
