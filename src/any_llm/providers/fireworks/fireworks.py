import os
from typing import Any

try:
    from fireworks import LLM
except ImportError:
    msg = "fireworks-ai is not installed. Please install it with `pip install any-llm-sdk[fireworks]`"
    raise ImportError(msg)

from pydantic import BaseModel
from any_llm.types import Stream
from any_llm.types import ChatCompletionChunk
from any_llm.types import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.providers.helpers import create_completion_from_response


class FireworksProvider(Provider):
    """
    Fireworks AI Provider using the native fireworks-ai client.

    This provider uses the fireworks-ai SDK to communicate with Fireworks AI's API,
    properly handling their JSON schema format for structured outputs.
    """

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Fireworks provider."""
        if not config.api_key:
            config.api_key = os.getenv("FIREWORKS_API_KEY")
        if not config.api_key:
            raise MissingApiKeyError("Fireworks", "FIREWORKS_API_KEY")

        # Store the API key for client creation
        self.api_key = config.api_key

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Fireworks."""

        if kwargs.get("stream", False) is True:
            raise UnsupportedParameterError("stream", "Fireworks")

        # Initialize the LLM client with the model
        llm = LLM(
            model=model,
            deployment_type="serverless",
            api_key=self.api_key,
        )

        # Handle response_format for Pydantic models
        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # Convert Pydantic model to Fireworks JSON schema format
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": response_format.__name__, "schema": response_format.model_json_schema()},
                }
            else:
                # response_format is already a dict, pass it through
                kwargs["response_format"] = response_format

        # Make the API call
        response = llm.chat.completions.create(
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        # Convert to OpenAI format using the new utility
        return create_completion_from_response(
            response_data=response.model_dump(),
            provider_name="Fireworks",
            model=model,
        )
