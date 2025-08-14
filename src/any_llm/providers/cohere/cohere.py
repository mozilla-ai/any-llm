from collections.abc import Iterator
from typing import Any

try:
    import cohere

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig, Provider
from any_llm.providers.cohere.utils import (
    _convert_response,
    _create_openai_chunk_from_cohere_chunk,
)
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


class CohereProvider(Provider):
    """Cohere Provider using the new response conversion utilities."""

    PROVIDER_NAME = "cohere"
    ENV_API_KEY_NAME = "CO_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://cohere.com/api"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = False

    PACKAGES_INSTALLED = PACKAGES_INSTALLED

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

    def completion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        """Create a chat completion using Cohere."""
        if params.response_format is not None:
            msg = "response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)
        if params.stream and params.response_format is not None:
            msg = "stream and response_format"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)
        if params.parallel_tool_calls is not None:
            msg = "parallel_tool_calls"
            raise UnsupportedParameterError(msg, self.PROVIDER_NAME)

        if params.stream:
            return self._stream_completion(
                params.model_id,
                params.messages,
                **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "response_format", "stream"}),
                **kwargs,
            )
        response = self.client.chat(
            model=params.model_id,
            messages=params.messages,  # type: ignore[arg-type]
            **params.model_dump(exclude_none=True, exclude={"model_id", "messages", "stream"}),
            **kwargs,
        )

        return _convert_response(response, params.model_id)
