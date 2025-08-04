from typing import Any, Iterator

try:
    import cohere
except ImportError:
    msg = "cohere is not installed. Please install it with `pip install any-llm-sdk[cohere]`"
    raise ImportError(msg)

from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion
from any_llm.provider import Provider, ApiConfig
from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.cohere.utils import (
    _create_openai_chunk_from_cohere_chunk,
    _convert_response,
)


class CohereProvider(Provider):
    """Cohere Provider using the new response conversion utilities."""

    PROVIDER_NAME = "Cohere"
    ENV_API_KEY_NAME = "CO_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://cohere.com/api"

    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDING = False

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Cohere provider."""
        super().__init__(config)
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

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Cohere provider."""
        if kwargs.get("response_format", None) is not None:
            raise UnsupportedParameterError("response_format", self.PROVIDER_NAME)
        if kwargs.get("stream", False) and kwargs.get("response_format", None) is not None:
            raise UnsupportedParameterError("stream and response_format", self.PROVIDER_NAME)
        if kwargs.get("parallel_tool_calls", None) is not None:
            raise UnsupportedParameterError("parallel_tool_calls", self.PROVIDER_NAME)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Cohere."""

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
