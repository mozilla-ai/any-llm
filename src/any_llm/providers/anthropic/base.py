from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from any_llm.any_llm import AnyLLM

MISSING_PACKAGES_ERROR = None
try:
    from anthropic import AsyncAnthropic

    from .utils import (
        _convert_models_list,
        _convert_params,
        _convert_response,
        _create_openai_chunk_from_anthropic_chunk,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from anthropic import AsyncAnthropic, AsyncAnthropicVertex
    from anthropic.types import Message
    from anthropic.types.model_info import ModelInfo as AnthropicModelInfo

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model


class BaseAnthropicProvider(AnyLLM, ABC):
    """
    Base provider for Anthropic-compatible services.

    This class provides a common foundation for providers that use Anthropic-compatible APIs.
    Subclasses need to override `_init_client()` for provider-specific client initialization.
    """

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = False
    SUPPORTS_BATCH = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncAnthropic | AsyncAnthropicVertex

    @abstractmethod
    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Anthropic API."""
        return _convert_params(params, **kwargs)

    @staticmethod
    @override
    def _convert_completion_response(response: Message) -> ChatCompletion:
        """Convert Anthropic Message to OpenAI ChatCompletion format."""
        return _convert_response(response)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Anthropic streaming chunk to OpenAI ChatCompletionChunk format."""
        model_id = kwargs.get("model_id", "unknown")
        return _create_openai_chunk_from_anthropic_chunk(response, model_id)

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Anthropic does not support embeddings."""
        msg = "Anthropic does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Anthropic does not support embeddings."""
        msg = "Anthropic does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_list_models_response(response: list[AnthropicModelInfo]) -> Sequence[Model]:
        """Convert Anthropic models list to OpenAI format."""
        return _convert_models_list(response)

    async def _stream_completion_async(self, **kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        async with self.client.messages.stream(**kwargs) as anthropic_stream:
            async for event in anthropic_stream:
                yield self._convert_completion_chunk_response(event, model_id=kwargs.get("model", "unknown"))

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        kwargs["provider_name"] = self.PROVIDER_NAME
        converted_kwargs = self._convert_completion_params(params, **kwargs)

        if converted_kwargs.pop("stream", False):
            return self._stream_completion_async(**converted_kwargs)

        message = await self.client.messages.create(**converted_kwargs)

        return self._convert_completion_response(message)
