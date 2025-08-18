from collections.abc import AsyncIterator, Iterator
from typing import Any

try:
    from voyageai.client import Client
    from voyageai.client_async import AsyncClient

    from any_llm.providers.voyage.utils import (
        _create_openai_embedding_response_from_voyage,
    )

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from any_llm.provider import Provider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse


class VoyageProvider(Provider):
    """
    Provider for Voyage AI services.
    """

    PROVIDER_NAME = "voyage"
    ENV_API_KEY_NAME = "VOYAGE_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.voyageai.com/"

    SUPPORTS_COMPLETION = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_STREAMING = False
    SUPPORTS_RESPONSES = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = False

    PACKAGES_INSTALLED = PACKAGES_INSTALLED

    def embedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        if isinstance(inputs, str):
            inputs = [inputs]

        client = Client(api_key=self.config.api_key)
        result = client.embed(
            texts=inputs,
            model=model,
            **kwargs,
        )
        return _create_openai_embedding_response_from_voyage(model, result)

    async def aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        if isinstance(inputs, str):
            inputs = [inputs]

        client = AsyncClient(api_key=self.config.api_key)
        result = await client.embed(
            texts=inputs,
            model=model,
            **kwargs,
        )
        return _create_openai_embedding_response_from_voyage(model, result)

    async def acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        msg = "voyage provider doesn't support completion."
        raise NotImplementedError(msg)

    def completion(self, params: CompletionParams, **kwargs: Any) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        msg = "voyage provider doesn't support completion."
        raise NotImplementedError(msg)
