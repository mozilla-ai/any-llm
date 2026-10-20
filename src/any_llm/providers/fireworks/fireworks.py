import inspect
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

import httpx
from openai import AsyncStream
from openresponses_types import ResponseResource
from typing_extensions import override

from any_llm.exceptions import UnsupportedParameterError
from any_llm.logging import logger
from any_llm.providers.openai.base import BaseOpenAIProvider, OpenAIChunkStream
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.types.rerank import RerankResponse
from any_llm.types.responses import Response, ResponsesParams, ResponseStreamEvent
from any_llm.utils.reasoning import strip_extra_content
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type

MISSING_PACKAGES_ERROR = None
try:
    from fireworks import AsyncFireworks

    from .utils import (
        convert_fireworks_chat_completion,
        convert_fireworks_chat_completion_chunk,
        convert_fireworks_rerank_response,
        extract_reasoning_from_response,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from fireworks._types import RequestOptions

# The OpenAI client's default. The SDK's own 60s read timeout would cut off long non-streaming
# reasoning calls. openai's DEFAULT_TIMEOUT cannot be reused: newer openai releases build it on
# httpx2, and the Fireworks SDK, which is on httpx, takes it for a number.
DEFAULT_TIMEOUT = httpx.Timeout(timeout=600.0, connect=5.0)


def _sdk_base_url(api_base: str) -> str:
    """Return the Fireworks SDK base URL for an OpenAI-compatible `api_base`.

    `api_base` must name the inference root ending in `/v1`, as the OpenAI client expects, because
    the SDK appends `/v1/...` to an overridden base URL itself. Any other `api_base` is passed on
    unchanged, so chat completions and rerank go to `<api_base>/v1/...` while Responses and model
    listing go to `<api_base>/...`; a warning is logged.
    """
    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        logger.warning(
            "Fireworks api_base %s does not end in /v1, so chat completions and rerank go to %s/v1/... "
            "while Responses and model listing go to %s/...",
            api_base,
            base,
            base,
        )
        return base
    return base.removesuffix("/v1")


def _sdk_client_kwargs(client_args: dict[str, Any]) -> dict[str, Any]:
    """Return the client arguments the Fireworks SDK constructor accepts.

    The provider builds an `AsyncFireworks` beside an `AsyncOpenAI` from one set of client
    arguments, and OpenAI-only ones such as `organization` or `project` would make the Fireworks
    constructor raise `TypeError` and break the whole provider.
    """
    accepted = inspect.signature(AsyncFireworks.__init__).parameters
    return {key: value for key, value in client_args.items() if key in accepted}


class FireworksProvider(BaseOpenAIProvider):
    """Fireworks provider.

    Chat completions and rerank go through the Fireworks SDK. The SDK has no Responses or
    serverless model-listing resource, so those stay on the OpenAI-compatible client.
    """

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

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    fireworks_client: "AsyncFireworks"

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        super()._init_client(api_key=api_key, api_base=api_base, **kwargs)
        self.fireworks_client = AsyncFireworks(
            api_key=api_key,
            base_url=_sdk_base_url(api_base or self.API_BASE),
            **{"timeout": DEFAULT_TIMEOUT, **_sdk_client_kwargs(kwargs)},
        )

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for the Fireworks SDK.

        The SDK has no `parse()` helper, so every structured-output type is sent as a JSON schema
        and the base layer parses the reply. It also has no `stream_options` argument, though the
        API takes one, so that goes in `extra_body`.
        """
        converted_params = BaseOpenAIProvider._convert_completion_params(params, **kwargs)
        if is_structured_output_type(params.response_format):
            converted_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": params.response_format.__name__,
                    "schema": get_json_schema(params.response_format),
                },
            }
        stream_options = converted_params.pop("stream_options", None)
        if stream_options is not None:
            converted_params["extra_body"] = {
                **(converted_params.get("extra_body") or {}),
                "stream_options": stream_options,
            }
        return converted_params

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert a Fireworks SDK chat completion to a ChatCompletion."""
        return convert_fireworks_chat_completion(response)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert a Fireworks SDK chat completion chunk to a ChatCompletionChunk."""
        return convert_fireworks_chat_completion_chunk(response)

    @override
    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if params.reasoning_effort == "auto":
            params.reasoning_effort = self._DEFAULT_REASONING_EFFORT
        params.messages = strip_extra_content(params.messages)

        completion_kwargs = self._convert_completion_params(params, **kwargs)
        response = await self.fireworks_client.chat.completions.create(
            model=params.model_id,
            messages=cast("Any", params.messages),
            **completion_kwargs,
        )
        if params.stream:
            # OpenAIChunkStream only iterates and closes the stream, which the Fireworks SDK stream supports.
            return OpenAIChunkStream(cast("Any", response), self._convert_completion_chunk_response)
        return self._convert_completion_response(response)

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
        # The SDK has no rerank resource, so post through its client to keep the provider's auth,
        # base URL, retries and error mapping. A per-request timeout is a request option there,
        # not a body field.
        timeout = kwargs.pop("timeout", None)
        options: RequestOptions = {"timeout": timeout} if timeout is not None else {}
        response = await self.fireworks_client.post(
            "/v1/rerank",
            body=self._convert_rerank_params(model, query, documents, **kwargs),
            cast_to=dict[str, Any],
            options=options,
        )
        return self._convert_rerank_response(response)
