from typing import Any

try:
    from llama_api_client import LlamaClient
except ImportError:
    msg = "llama-api-client is not installed. Please install it with `pip install llama-api-client`"
    raise ImportError(msg)

from pydantic import BaseModel

from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider
from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.helpers import (
    create_completion_from_response,
)
from .utils import convert_to_llama_format

class LlamaProvider(Provider):
    """Llama Provider using the new response conversion utilities."""

    PROVIDER_NAME = "Llama"
    ENV_API_KEY_NAME = "LLAMA_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://www.llama.com/docs/integration-guides/"

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Llama provider."""
        if kwargs.get("stream", False):
            raise UnsupportedParameterError("stream", self.PROVIDER_NAME)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Llama."""
        client = LlamaClient(api_key=self.config.api_key, timeout=kwargs.get("timeout", None))

        # Convert max_tokens to max_new_tokens (Llama specific)
        if "max_tokens" in kwargs:
            kwargs["max_new_tokens"] = kwargs.pop("max_tokens")

        # Handle response_format for Pydantic models
        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                # Convert Pydantic model to Llama JSON format
                messages = convert_to_llama_format(messages, response_format=response_format)

        # Make the API call
        response = client.chat_completion(
            model=model,
            messages=messages,
            **kwargs,
        )

        # Convert to OpenAI format using the new utility
        return create_completion_from_response(
            response_data=response,
            model=model,
            provider_name=self.PROVIDER_NAME,
        )
