from typing import Any

try:
    from voyageai.client import Client
except ImportError:
    msg = "voyageai is not installed. Please install it with `pip install any-llm-sdk[voyage]`"
    raise ImportError(msg)

from openai._streaming import Stream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from openai.types.create_embedding_response import CreateEmbeddingResponse

from any_llm.provider import Provider
from any_llm.providers.voyage.utils import (
    _create_openai_embedding_response_from_voyage,
)


class VoyageProvider(Provider):
    """
    Provider for Voyage AI services.
    """

    PROVIDER_NAME = "Voyage"
    ENV_API_KEY_NAME = "VOYAGE_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.voyageai.com/"

    SUPPORTS_COMPLETION = False
    SUPPORTS_STREAMING = False
    SUPPORTS_EMBEDDING = True

    def verify_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Default is that all kwargs are supported."""
        pass

    def _make_api_call(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ChatCompletion | Stream[ChatCompletionChunk]:
        raise NotImplementedError("voyage provider doesn't support completion.")

    def embedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        if not self.SUPPORTS_EMBEDDING:
            raise NotImplementedError("This provider does not support embeddings.")

        if isinstance(inputs, str):
            inputs = [inputs]

        client = Client(api_key=self.config.api_key)
        result = client.embed(
            texts=inputs,
            model=model,
            **kwargs,
        )
        return _create_openai_embedding_response_from_voyage(model, result)
