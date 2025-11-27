from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from httpx import AsyncClient

from any_llm.any_llm import AnyLLM
from any_llm.logging import logger
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionParams,
    CompletionUsage,
    CreateEmbeddingResponse,
)

from .utils import get_provider_key, post_completion_usage_event

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from any_llm.types.model import Model


class PlatformProvider(AnyLLM):
    PROVIDER_NAME = "platform"
    ENV_API_KEY_NAME = "ANY_LLM_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/mozilla-ai/any-llm"

    # All features are marked as supported, but depending on which provider you call inside the gateway, they may not all work.
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = True

    def __init__(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any):
        self.any_llm_key = self._verify_and_set_api_key(api_key)
        self.api_base = api_base
        self.kwargs = kwargs

        self._init_client(api_key=api_key, api_base=api_base, **kwargs)

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = AsyncClient(**kwargs)

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        raise NotImplementedError

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        raise NotImplementedError

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        raise NotImplementedError

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        raise NotImplementedError

    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        completion = await self.provider._acompletion(params=params, **kwargs)

        if not params.stream:
            await post_completion_usage_event(
                client=self.client,
                any_llm_key=self.any_llm_key,  # type: ignore[arg-type]
                provider=self.provider.PROVIDER_NAME,
                completion=cast("ChatCompletion", completion),
            )
            return completion

        # For streaming, wrap the iterator to collect usage info
        return self._stream_with_usage_tracking(cast("AsyncIterator[ChatCompletionChunk]", completion))

    async def _stream_with_usage_tracking(
        self, stream: AsyncIterator[ChatCompletionChunk]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Wrap the stream to track usage after completion."""
        chunks: list[ChatCompletionChunk] = []

        async for chunk in stream:
            chunks.append(chunk)
            yield chunk

        # After stream completes, reconstruct completion for usage tracking
        if chunks:
            # Combine chunks into a single ChatCompletion-like object
            final_completion = self._combine_chunks(chunks)
            await post_completion_usage_event(
                client=self.client,
                any_llm_key=self.any_llm_key,  # type: ignore [arg-type]
                provider=self.provider.PROVIDER_NAME,
                completion=final_completion,
            )

    def _combine_chunks(self, chunks: list[ChatCompletionChunk]) -> ChatCompletion:
        """Combine streaming chunks into a ChatCompletion for usage tracking."""
        # Get the last chunk which typically has the full usage info
        last_chunk = chunks[-1]

        if not last_chunk.usage:
            msg = (
                "The last chunk of your streaming response does not contain usage data. "
                "Consult your provider documentation on how to retrieve it."
            )
            logger.error(msg)

            return ChatCompletion(
                id=last_chunk.id,
                model=last_chunk.model,
                created=last_chunk.created,
                object="chat.completion",
                usage=CompletionUsage(
                    completion_tokens=0,
                    prompt_tokens=0,
                    total_tokens=0,
                ),
                choices=[],
            )

        # Create a minimal ChatCompletion object with the data needed for usage tracking
        # We only need id, model, created, usage, and object type
        return ChatCompletion(
            id=last_chunk.id,
            model=last_chunk.model,
            created=last_chunk.created,
            object="chat.completion",
            usage=last_chunk.usage if hasattr(last_chunk, "usage") and last_chunk.usage else None,
            choices=[],
        )

    @property
    def provider(self) -> AnyLLM:
        return self._provider

    @provider.setter
    def provider(self, provider_class: type[AnyLLM]) -> None:
        provider_key = get_provider_key(any_llm_key=self.any_llm_key, provider=provider_class)  # type: ignore[arg-type]
        self._provider = provider_class(api_key=provider_key, api_base=self.api_base, **self.kwargs)
