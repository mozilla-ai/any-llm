from collections.abc import Iterator
from typing import Any

try:
    from voyageai.client import Client

    from any_llm.providers.voyage.utils import (
        _create_openai_embedding_response_from_voyage,
    )

    PACKAGES_INSTALLED = True
except ImportError:
    PACKAGES_INSTALLED = False

from any_llm.provider import Provider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CreateEmbeddingResponse


class VoyageProvider(Provider):
    """
    Provider for Voyage AI services.
    """

    PROVIDER_NAME = "Voyage"
    ENV_API_KEY_NAME = "VOYAGE_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.voyageai.com/"

    SUPPORTS_COMPLETION = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_STREAMING = False
    SUPPORTS_RESPONSES = False
    SUPPORTS_EMBEDDING = True

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

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        msg = "voyage provider doesn't support completion."
        raise NotImplementedError(msg)
