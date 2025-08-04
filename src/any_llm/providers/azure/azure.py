import os
import json
import urllib.request
import urllib.error
from typing import Any

from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import UnsupportedParameterError
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from any_llm.providers.azure.utils import _convert_response


class AzureProvider(Provider):
    """Azure Provider using urllib for direct API calls."""

    PROVIDER_NAME = "Azure"
    ENV_API_KEY_NAME = "AZURE_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://azure.microsoft.com/en-us/products/ai-services/openai-service"

    SUPPORTS_STREAMING = False

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Azure provider."""
        super().__init__(config)
        self.base_url = config.api_base or os.getenv("AZURE_BASE_URL")
        self.api_version = os.getenv("AZURE_API_VERSION")

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Azure provider."""
        if kwargs.get("stream", False):
            raise UnsupportedParameterError("stream", self.PROVIDER_NAME)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Azure."""
        if not self.base_url:
            raise ValueError(
                "For Azure, base_url is required. Check your deployment page for a URL like this - "
                "https://<model-deployment-name>.<region>.models.ai.azure.com"
            )

        url = f"{self.base_url}/chat/completions"
        if self.api_version:
            url = f"{url}?api-version={self.api_version}"

        data = {"messages": messages, **kwargs}

        body = json.dumps(data).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.config.api_key or "",
        }

        req = urllib.request.Request(url, body, headers)
        with urllib.request.urlopen(req) as response:
            result = response.read()
            response_data = json.loads(result)

            return _convert_response(response_data)
