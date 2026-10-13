from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from openai import AsyncStream
from openresponses_types import ResponseResource
from typing_extensions import override

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.rerank import RerankResponse
from any_llm.types.responses import Response, ResponsesParams, ResponseStreamEvent

from .utils import convert_fireworks_rerank_response, extract_reasoning_from_response

if TYPE_CHECKING:
    from openai._types import RequestOptions


class FireworksProvider(BaseOpenAIProvider):
    PROVIDER_NAME = "fireworks"
    ENV_API_KEY_NAME = "FIREWORKS_API_KEY"
    ENV_API_BASE_NAME = "FIREWORKS_API_BASE"
    PROVIDER_DOCUMENTATION_URL = "https://fireworks.ai/api"
    API_BASE = "https://api.fireworks.ai/inference/v1"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_RERANK = True

    @override
    async def _aresponses(
        self, params: ResponsesParams, **kwargs: Any
    ) -> ResponseResource | Response | AsyncIterator[ResponseStreamEvent]:
        """Call Fireworks Responses API and normalize into ChatCompletion/Chunks."""
        response = await super()._aresponses(params, **kwargs)
        if isinstance(response, Response) and not isinstance(response, AsyncStream):
            return extract_reasoning_from_response(response)
        return response

    @staticmethod
    @override
    def _convert_rerank_params(model: str, query: str, documents: list[str], **kwargs: Any) -> dict[str, Any]:
        """Convert rerank parameters for the Fireworks API.

        `return_documents` defaults to False: Fireworks echoes every document back by default, and
        RerankResponse carries no document text.

        Raises:
            UnsupportedParameterError: If `max_tokens_per_doc` is provided.
        """
        if kwargs.get("max_tokens_per_doc") is not None:
            msg = "max_tokens_per_doc"
            raise UnsupportedParameterError(
                msg,
                FireworksProvider.PROVIDER_NAME,
                "Fireworks rerank has no per-document token limit.",
            )

        params: dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": documents,
            "return_documents": False,
        }
        if kwargs.get("top_n") is not None:
            params["top_n"] = kwargs["top_n"]
        for key in ("return_documents", "task"):
            if key in kwargs:
                params[key] = kwargs[key]
        return params

    @staticmethod
    @override
    def _convert_rerank_response(response: Any) -> RerankResponse:
        """Convert a Fireworks rerank response to a normalized RerankResponse."""
        return convert_fireworks_rerank_response(response)

    @override
    async def _arerank(
        self,
        model: str,
        query: str,
        documents: list[str],
        **kwargs: Any,
    ) -> RerankResponse:
        """Rerank `documents` against `query` with the Fireworks `/rerank` endpoint."""
        # The OpenAI SDK has no rerank resource, so post through its client to keep the
        # provider's auth, base URL, retries and error mapping. A per-request timeout is a
        # request option there, not a body field.
        timeout = kwargs.pop("timeout", None)
        options: RequestOptions = {"timeout": timeout} if timeout is not None else {}
        response = await self.client.post(
            "/rerank",
            body=self._convert_rerank_params(model, query, documents, **kwargs),
            cast_to=dict[str, Any],
            options=options,
        )
        return self._convert_rerank_response(response)
