from typing import Any, cast, Iterator

try:
    import cerebras.cloud.sdk as cerebras
    from cerebras.cloud.sdk.types.chat.chat_completion import ChatChunkResponse
    import instructor
except ImportError:
    msg = "cerebras or instructor is not installed. Please install it with `pip install any-llm-sdk[cerebras]`"
    raise ImportError(msg)

from openai.types.chat.chat_completion import ChatCompletion
from openai._streaming import Stream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from any_llm.provider import Provider, ApiConfig, convert_instructor_response
from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.cerebras.utils import (
    _create_openai_chunk_from_cerebras_chunk,
    _convert_response,
)


class CerebrasProvider(Provider):
    """Cerebras Provider using the official Cerebras SDK with instructor support for structured outputs."""

    PROVIDER_NAME = "Cerebras"
    ENV_API_KEY_NAME = "CEREBRAS_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.cerebras.ai/"

    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDING = False

    def __init__(self, config: ApiConfig) -> None:
        """Initialize Cerebras provider."""
        super().__init__(config)
        self.client = cerebras.Cerebras(api_key=config.api_key)
        # Create instructor client for structured output support
        self.instructor_client = instructor.from_cerebras(self.client)

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Cerebras provider."""
        if kwargs.get("stream", False) and kwargs.get("response_format", None) is not None:
            raise UnsupportedParameterError("stream and response_format", self.PROVIDER_NAME)

    def _stream_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        # Get the Cerebras stream
        cerebras_stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            **kwargs,
        )

        for chunk in cerebras_stream:
            # Only process ChatChunkResponse objects
            if isinstance(chunk, ChatChunkResponse):
                yield _create_openai_chunk_from_cerebras_chunk(chunk)
            else:
                raise ValueError(f"Unsupported chunk type: {type(chunk)}")

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """Create a chat completion using Cerebras with instructor support for structured outputs."""

        # Handle response_format for structured output
        if "response_format" in kwargs:
            response_format = kwargs.pop("response_format")
            # Use instructor for structured output
            instructor_response = self.instructor_client.chat.completions.create(
                model=model,
                messages=cast(Any, messages),
                response_model=response_format,
                **kwargs,
            )

            # Convert instructor response to ChatCompletion format
            return convert_instructor_response(instructor_response, model, self.PROVIDER_NAME)

        if kwargs.get("stream", False):
            # Remove stream parameter before passing to streaming method
            kwargs.pop("stream")
            # Return the streaming generator
            return self._stream_completion(model, messages, **kwargs)  # type: ignore[return-value]
        else:
            # Use regular create method for non-structured outputs
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )

            # Convert response to dict format for processing
            # Handle the case where response might be a Stream object
            if hasattr(response, "model_dump"):
                response_data = response.model_dump()
            else:
                # If it's a streaming response, we need to handle it differently
                raise ValueError("Streaming responses are not supported in this context")

            # Convert to OpenAI format
            return _convert_response(response_data)
