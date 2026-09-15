from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import UnsupportedParameterError

MISSING_PACKAGES_ERROR: ImportError | None = None
PYTHON_VERSION_INCOMPATIBLE = sys.version_info >= (3, 14)

if PYTHON_VERSION_INCOMPATIBLE:
    MISSING_PACKAGES_ERROR = ImportError(
        "The 'voyageai' package is not compatible with Python 3.14+. "
        "The package uses pydantic v1 which has breaking changes in Python 3.14. "
        "Please use Python 3.13 or earlier to use this provider, or wait for an updated voyageai package."
    )
else:
    try:
        from voyageai.client_async import AsyncClient

        from .utils import (
            _convert_voyage_rerank_response,
            _create_openai_embedding_response_from_voyage,
        )
    except ImportError as e:
        MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import Sequence

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model
    from any_llm.types.rerank import RerankResponse


class VoyageProvider(AnyLLM):
    """
    Provider for Voyage AI services.
    """

    PROVIDER_NAME = "voyage"
    ENV_API_KEY_NAME = "VOYAGE_API_KEY"
    ENV_API_BASE_NAME = "VOYAGE_API_BASE"
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
    SUPPORTS_RERANK = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncClient

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Voyage API."""
        msg = "Voyage does not support completions"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert Voyage response to OpenAI format."""
        msg = "Voyage does not support completions"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Voyage chunk response to OpenAI format."""
        msg = "Voyage does not support completions"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Convert embedding parameters for Voyage."""
        if isinstance(params, str):
            params = [params]
        converted_params = {"texts": params}
        converted_params.update(kwargs)
        return converted_params

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Convert Voyage embedding response to OpenAI format."""
        # We need the model parameter for conversion
        model = response.get("model", "voyage-model")
        return _create_openai_embedding_response_from_voyage(model, response["result"])

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Voyage list models response to OpenAI format."""
        msg = "Voyage does not support listing models"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_rerank_params(model: str, query: str, documents: list[str], **kwargs: Any) -> dict[str, Any]:
        """Convert rerank parameters for the Voyage API.

        Voyage names the result limit `top_k` rather than `top_n`.

        Raises:
            UnsupportedParameterError: If `max_tokens_per_doc` is provided.
        """
        if kwargs.get("max_tokens_per_doc") is not None:
            msg = "max_tokens_per_doc"
            raise UnsupportedParameterError(
                msg,
                "voyage",
                "Voyage only exposes a boolean `truncation` flag, not a per-document token limit.",
            )

        params: dict[str, Any] = {
            "query": query,
            "documents": documents,
        }
        if kwargs.get("top_n") is not None:
            params["top_k"] = kwargs["top_n"]
        for key in ("truncation",):
            if key in kwargs:
                params[key] = kwargs[key]
        return params

    @staticmethod
    @override
    def _convert_rerank_response(response: Any) -> RerankResponse:
        """Convert a Voyage rerank response to a normalized RerankResponse."""
        return _convert_voyage_rerank_response(response)

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = AsyncClient(api_key=api_key, **kwargs)

    @override
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

    @override
    async def _arerank(
        self,
        model: str,
        query: str,
        documents: list[str],
        **kwargs: Any,
    ) -> RerankResponse:
        rerank_kwargs = self._convert_rerank_params(model, query, documents, **kwargs)
        result = await self.client.rerank(
            model=model,
            **rerank_kwargs,
        )
        return self._convert_rerank_response(result)
