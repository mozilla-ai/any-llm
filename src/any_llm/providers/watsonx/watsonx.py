import os
from typing import Any

try:
    from ibm_watsonx_ai import Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference  # type: ignore[attr-defined]
except ImportError:
    msg = "ibm-watsonx-ai is not installed. Please install it with `pip install any-llm-sdk[watsonx]`"
    raise ImportError(msg)

from any_llm.types import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError
from any_llm.types import Stream
from any_llm.types import ChatCompletionChunk
from any_llm.providers.watsonx.utils import _convert_response


class WatsonxProvider(Provider):
    """IBM Watsonx Provider using the official IBM Watsonx AI SDK."""

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Watsonx provider."""
        # Get configuration from config or environment variables
        self.service_url = config.api_base or os.getenv("WATSONX_SERVICE_URL")
        self.api_key = config.api_key or os.getenv("WATSONX_API_KEY")
        self.project_id = os.getenv("WATSONX_PROJECT_ID")

        # Only validate API key during instantiation for consistency with other providers
        if not self.api_key:
            raise MissingApiKeyError("Watsonx", "WATSONX_API_KEY")

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Watsonx."""
        # Validate required configuration at runtime
        if not self.service_url:
            raise ValueError(
                "Missing WatsonX service URL. Please provide it in the config or set the WATSONX_SERVICE_URL environment variable."
            )
        if not self.project_id:
            raise ValueError(
                "Missing WatsonX project ID. Please provide it in the config or set the WATSONX_PROJECT_ID environment variable."
            )

        # Create ModelInference instance
        model_inference = ModelInference(
            model_id=model,
            credentials=Credentials(
                api_key=self.api_key,
                url=self.service_url,
            ),
            project_id=self.project_id,
        )

        # Make the API call
        response = model_inference.chat(
            messages=messages,
            params=kwargs,
        )

        # Convert to OpenAI format
        return _convert_response(response)
