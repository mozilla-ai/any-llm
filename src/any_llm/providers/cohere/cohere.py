from typing import Any, Iterator

try:
    import cohere
except ImportError:
    msg = "cohere is not installed. Please install it with `pip install any-llm-sdk[cohere]`"
    raise ImportError(msg)

from any_llm.types.completion import ChatCompletionChunk, ChatCompletion
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
    SUPPORTS_COMPLETION = True
    SUPPORTS_REASONING = False
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
        cohere_stream = self.client.chat_stream(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )

        for chunk in cohere_stream:
            yield _create_openai_chunk_from_cohere_chunk(chunk)

    @classmethod
    def verify_kwargs(cls, kwargs: dict[str, Any]) -> None:
        """Verify the kwargs for the Cohere provider."""
        if kwargs.get("response_format", None) is not None:
            raise UnsupportedParameterError("response_format", cls.PROVIDER_NAME)
        if kwargs.get("stream", False) and kwargs.get("response_format", None) is not None:
            raise UnsupportedParameterError("stream and response_format", cls.PROVIDER_NAME)
        if kwargs.get("parallel_tool_calls", None) is not None:
            raise UnsupportedParameterError("parallel_tool_calls", cls.PROVIDER_NAME)

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using Cohere."""

        if kwargs.get("stream", False):
            # Remove stream parameter before passing to streaming method
            kwargs.pop("stream")
            return self._stream_completion(model, messages, **kwargs)
        else:
            # Make the API call for non-streaming
            response = self.client.chat(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )

            return _convert_response(response, model)
