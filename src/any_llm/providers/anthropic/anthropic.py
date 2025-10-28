from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

    from anthropic.types import Message
    from anthropic.types.model_info import ModelInfo as AnthropicModelInfo

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model


class AnthropicProvider(AnyLLM):
    """
    Anthropic Provider using enhanced Provider framework.

    Handles conversion between OpenAI format and Anthropic's native format.
    """

    PROVIDER_NAME = "anthropic"
    ENV_API_KEY_NAME = "ANTHROPIC_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.anthropic.com/en/home"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncAnthropic

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=api_base,
            **kwargs,
        )

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Anthropic API."""
        return _convert_params(params, **kwargs)

    @staticmethod
    def _convert_completion_response(response: Message) -> ChatCompletion:
        """Convert Anthropic Message to OpenAI ChatCompletion format."""
        return _convert_response(response)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Anthropic streaming chunk to OpenAI ChatCompletionChunk format."""
        model_id = kwargs.get("model_id", "unknown")
        return _create_openai_chunk_from_anthropic_chunk(response, model_id)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Anthropic does not support embeddings."""
        msg = "Anthropic does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Anthropic does not support embeddings."""
        msg = "Anthropic does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_list_models_response(response: list[AnthropicModelInfo]) -> Sequence[Model]:
        """Convert Anthropic models list to OpenAI format."""
        return _convert_models_list(response)

    async def _stream_completion_async(self, **kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        async with self.client.messages.stream(
            **kwargs,
        ) as anthropic_stream:
            async for event in anthropic_stream:
                yield self._convert_completion_chunk_response(event, model_id=kwargs.get("model", "unknown"))

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

    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        models_list = await self.client.models.list(**kwargs)
        return self._convert_list_models_response(models_list.data)
