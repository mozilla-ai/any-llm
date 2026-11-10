from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from typing_extensions import override

from any_llm.exceptions import ProviderError, UnsupportedParameterError
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.openrouter.utils import _convert_models_list, build_reasoning_directive, convert_rerank_response
from any_llm.types.completion import CompletionParams
from any_llm.types.model import Model
from any_llm.types.rerank import RerankResponse

if TYPE_CHECKING:
    from openai._types import RequestOptions


class OpenrouterProvider(BaseOpenAIProvider):
    API_BASE = "https://openrouter.ai/api/v1"
    ENV_API_KEY_NAME = "OPENROUTER_API_KEY"
    ENV_API_BASE_NAME = "OPENROUTER_API_BASE"
    PROVIDER_NAME = "openrouter"
    PROVIDER_DOCUMENTATION_URL = "https://openrouter.ai/docs"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    # OpenRouter serves a native, OpenAI-compatible /responses endpoint, so the inherited
    # BaseOpenAIProvider._aresponses passthrough works as-is (no override needed here). One
    # caveat: OpenRouter's Responses API is stateless, so it rejects requests where the caller
    # explicitly passes store=True or previous_response_id. Those fields default to None and are
    # omitted from the request otherwise, so this only surfaces for callers who opt into them.
    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_EMBEDDING = True
    # OpenRouter does not expose a moderation endpoint.
    SUPPORTS_MODERATION = False
    SUPPORTS_DECIDE = True
    # The alpha decisions endpoint sits outside /v1: https://openrouter.ai/api/alpha/decisions.
    DECIDE_PATH = "../alpha/decisions"
    SUPPORTS_RERANK = True

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert OpenRouter list models response to valid Model objects."""
        return _convert_models_list(response)

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for OpenRouter API, including reasoning directive."""
        converted_params = BaseOpenAIProvider._convert_completion_params(params, **kwargs)

        reasoning_directive = build_reasoning_directive(
            reasoning=kwargs.get("reasoning"),
            reasoning_effort=params.reasoning_effort,
        )

        if reasoning_directive is not None:
            converted_params.pop("reasoning_effort", None)
            extra_body = converted_params.get("extra_body", {}).copy()
            extra_body["reasoning"] = reasoning_directive
            converted_params["extra_body"] = extra_body

        return converted_params

    @override
    async def _arerank(self, model: str, query: str, documents: list[str], **kwargs: Any) -> RerankResponse:
        """Rerank `documents` through OpenRouter's `/rerank` endpoint."""
        if kwargs.pop("max_tokens_per_doc", None) is not None:
            msg = "max_tokens_per_doc"
            raise UnsupportedParameterError(
                msg, self.PROVIDER_NAME, "OpenRouter's rerank endpoint has no per-document truncation limit."
            )
        # The OpenAI SDK has no rerank resource, so post through its client to keep the
        # provider's auth, retries and error mapping, as _adecide does.
        options: RequestOptions = {}
        if (timeout := kwargs.pop("timeout", None)) is not None:
            options["timeout"] = timeout
        if (extra_headers := kwargs.pop("extra_headers", None)) is not None:
            options["headers"] = extra_headers
        extra_body = kwargs.pop("extra_body", None) or {}
        body = {"model": model, "query": query, "documents": documents, **kwargs, **extra_body}
        response = await self.client.post("/rerank", body=body, cast_to=dict[str, Any], options=options)
        try:
            return convert_rerank_response(response)
        except (KeyError, TypeError, ValidationError) as error:
            msg = f"{self.PROVIDER_NAME} returned a rerank response that is not valid"
            raise ProviderError(msg, provider_name=self.PROVIDER_NAME) from error
