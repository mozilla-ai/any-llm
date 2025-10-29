from __future__ import annotations

from typing import TYPE_CHECKING, Any

from any_llm.any_llm import AnyLLM

MISSING_PACKAGES_ERROR = None
try:
    from voyageai.client_async import AsyncClient

    from .utils import (
        _create_openai_embedding_response_from_voyage,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import Sequence

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model


class VoyageProvider(AnyLLM):
    """
    Provider for Voyage AI services.
    """

    PROVIDER_NAME = "voyage"
    ENV_API_KEY_NAME = "VOYAGE_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.voyageai.com/"

    SUPPORTS_COMPLETION = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_STREAMING = False
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_RESPONSES = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = False
    SUPPORTS_BATCH = False

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncClient

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Voyage API."""
        msg = "Voyage does not support completions"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert Voyage response to OpenAI format."""
        msg = "Voyage does not support completions"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Voyage chunk response to OpenAI format."""
        msg = "Voyage does not support completions"
        raise NotImplementedError(msg)

    @staticmethod
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Voyage."""
        if isinstance(params, str):
            params = [params]
        converted_params = {"texts": params}
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Voyage embedding response to OpenAI format."""
        # We need the model parameter for conversion
        model = response.get("model", "voyage-model")
        return _create_openai_embedding_response_from_voyage(model, response["result"])

    @staticmethod
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Voyage list models response to OpenAI format."""
        msg = "Voyage does not support listing models"
        raise NotImplementedError(msg)

    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = AsyncClient(api_key=api_key, **kwargs)

    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        embedding_kwargs = self._convert_embedding_params(inputs, **kwargs)

        result = await self.client.embed(
            model=model,
            **embedding_kwargs,
        )
        response_data = {"model": model, "result": result}
        return self._convert_embedding_response(response_data)
