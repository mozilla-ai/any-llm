import os
from typing import Any, Iterator

try:
    import cohere
except ImportError:
    msg = "cohere is not installed. Please install it with `pip install any-llm-sdk[cohere]`"
    raise ImportError(msg)

from any_llm.types import Stream
from any_llm.types import ChatCompletionChunk
from any_llm.types import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.providers.cohere.utils import (
    _create_openai_chunk_from_cohere_chunk,
    _convert_response,
)


class CohereProvider(Provider):
    """Cohere Provider using the new response conversion utilities."""

    PROVIDER_NAME = "Cohere"

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Cohere provider."""
        if not config.api_key:
            config.api_key = os.getenv("CO_API_KEY")
        if not config.api_key:
            raise MissingApiKeyError("Cohere", "CO_API_KEY")

        self.client = cohere.ClientV2(api_key=config.api_key)

    def _stream_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        # Get the Cohere stream
        cohere_stream = self.client.chat_stream(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        for chunk in cohere_stream:
            yield _create_openai_chunk_from_cohere_chunk(chunk)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Cohere."""

        # Remove unsupported parameters
        if "response_format" in kwargs:
            if kwargs.get("stream", False):
                raise UnsupportedParameterError("response_format with streaming", self.PROVIDER_NAME)
            else:
                raise UnsupportedParameterError("response_format", self.PROVIDER_NAME)
        elif "parallel_tool_calls" in kwargs:
            raise UnsupportedParameterError("parallel_tool_calls", self.PROVIDER_NAME)

        if kwargs.get("stream", False):
            # Remove stream parameter before passing to streaming method
            kwargs.pop("stream")
            # Return the streaming generator
            return self._stream_completion(model, messages, **kwargs)  # type: ignore[return-value]
        else:
            # Make the API call for non-streaming
            response = self.client.chat(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )

            # Convert to OpenAI format
            return _convert_response(response, model)
