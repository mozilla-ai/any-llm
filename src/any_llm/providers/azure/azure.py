import os
import json
import urllib.request
import urllib.error
from typing import Any

from any_llm.types import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError
from any_llm.types import Stream
from any_llm.types import ChatCompletionChunk
from any_llm.providers.azure.utils import _convert_response


class AzureProvider(Provider):
    """Azure Provider using urllib for direct API calls."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Azure provider."""
        self.base_url = config.api_base or os.getenv("AZURE_BASE_URL")
        self.api_key = config.api_key or os.getenv("AZURE_API_KEY")
        self.api_version = os.getenv("AZURE_API_VERSION")

        if not self.api_key:
            raise MissingApiKeyError("Azure", "AZURE_API_KEY")

    def completion(
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

        # Build the URL
        url = f"{self.base_url}/chat/completions"
        if self.api_version:
            url = f"{url}?api-version={self.api_version}"

        # Prepare the request payload
        data = {"messages": messages}

        # Add tools if provided
        if "tools" in kwargs:
            data["tools"] = kwargs["tools"]
            kwargs.pop("tools")

        # Add tool_choice if provided
        if "tool_choice" in kwargs:
            data["tool_choice"] = kwargs["tool_choice"]
            kwargs.pop("tool_choice")

        # Add remaining kwargs
        data.update(kwargs)

        # Prepare the request
        body = json.dumps(data).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.api_key or "",
        }

        # Make the request to Azure endpoint
        req = urllib.request.Request(url, body, headers)
        with urllib.request.urlopen(req) as response:
            result = response.read()
            response_data = json.loads(result)

            # Convert to OpenAI format
            return _convert_response(response_data)
