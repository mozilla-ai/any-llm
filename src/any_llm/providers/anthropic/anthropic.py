from typing import Any, Iterator


try:
    from anthropic import Anthropic
    import instructor
except ImportError:
    msg = "anthropic or instructor is not installed. Please install it with `pip install any-llm-sdk[anthropic]`"
    raise ImportError(msg)

from any_llm.exceptions import UnsupportedParameterError
from any_llm.types import ChatCompletion, ChatCompletionChunk, Stream
from any_llm.provider import ApiConfig, Provider, convert_instructor_response
from any_llm.providers.anthropic.utils import (
    _create_openai_chunk_from_anthropic_chunk,
    _convert_response,
    _convert_kwargs,
)


class AnthropicProvider(Provider):
    """
    Anthropic Provider using enhanced Provider framework.

    Handles conversion between OpenAI format and Anthropic's native format.
    """

    PROVIDER_NAME = "Anthropic"
    ENV_API_KEY_NAME = "ANTHROPIC_API_KEY"

    def _initialize_client(self, config: ApiConfig) -> None:
        """Initialize the Anthropic client."""
        self.client = Anthropic(api_key=config.api_key, base_url=config.api_base)

        # Create instructor client for structured output
        self.instructor_client = instructor.from_anthropic(self.client)

    def _stream_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        self._initialize_client(self.config)
        # Get the Anthropic stream
        with self.client.messages.stream(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        ) as anthropic_stream:
            for event in anthropic_stream:
                yield _create_openai_chunk_from_anthropic_chunk(event)

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Anthropic with instructor support."""
        self._initialize_client(self.config)
        # Handle response_format for structured output
        kwargs = _convert_kwargs(kwargs)

        if "response_format" in kwargs:
            if kwargs.get("stream", False):
                raise UnsupportedParameterError("response_format with streaming", "Anthropic")

            response_format = kwargs.pop("response_format")
            # Use instructor for structured output
            instructor_response = self.instructor_client.messages.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                response_model=response_format,
                **kwargs,
            )

            # Convert instructor response to ChatCompletion format
            return convert_instructor_response(instructor_response, model, "anthropic")

        if kwargs.get("stream", False):
            # Return the streaming generator
            kwargs.pop("stream")
            return self._stream_completion(model, messages, **kwargs)  # type: ignore[return-value]
        else:
            message = self.client.messages.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )
            return _convert_response(message)
