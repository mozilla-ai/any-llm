from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.utils.aio import aclose_quietly
from any_llm.utils.reasoning import strip_extra_content
from any_llm.utils.structured_output import get_json_schema, is_structured_output_type

MISSING_PACKAGES_ERROR = None
try:
    from sarvamai import AsyncSarvamAI, SarvamAIEnvironment
    from sarvamai.chat.client import AsyncChatClient

    from .utils import _convert_chunk, _convert_models_list, _convert_response

    # The keyword set of completions_v2 changes between SDK releases (0.1.46 has min_p, later
    # ones add logprobs), and the method rejects unknown keywords, so it is read from the SDK.
    _SDK_COMPLETION_PARAMS = frozenset(inspect.signature(AsyncChatClient.completions_v2).parameters) - {"self"}
except ImportError as e:
    MISSING_PACKAGES_ERROR = e
    _SDK_COMPLETION_PARAMS = frozenset()


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model


class SarvamProvider(AnyLLM):
    """Sarvam AI provider using the official ``sarvamai`` SDK and its v2 chat completions API."""

    PROVIDER_NAME = "sarvam"
    ENV_API_KEY_NAME = "SARVAM_API_KEY"
    ENV_API_BASE_NAME = "SARVAM_API_BASE"
    PROVIDER_DOCUMENTATION_URL = "https://docs.sarvam.ai/"
    # The SDK appends the versioned path (v2/chat/completions, v2/models) to this host.
    API_BASE = "https://api.sarvam.ai"

    SUPPORTS_COMPLETION = True
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_BATCH = False
    SUPPORTS_RERANK = False

    # The Fern-generated SDK takes timeout and retries per request in `request_options`, and its
    # client constructor has no retry setting, so both are carried into every call.
    TIMEOUT_SUPPORT = "mapped"
    MAX_RETRIES_SUPPORT = "mapped"

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncSarvamAI
    _max_retries: int | None = None

    @override
    def _apply_max_retries(self, max_retries: int, client_kwargs: dict[str, Any]) -> None:
        # _init_client takes it back out, since AsyncSarvamAI rejects the keyword.
        client_kwargs.setdefault("max_retries", max_retries)

    def _request_options(self, request_options: dict[str, Any] | None) -> dict[str, Any] | None:
        options = dict(request_options or {})
        if self._max_retries is not None:
            options.setdefault("max_retries", self._max_retries)
        return options or None

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        converted = params.model_dump(
            exclude_none=True,
            exclude={
                "model_id",
                "messages",
                "stream",
                "stream_options",
                "response_format",
                "max_completion_tokens",
            },
        )
        if converted.get("reasoning_effort") == "auto":
            converted.pop("reasoning_effort")
        # Sarvam documents only max_tokens, which also counts reasoning tokens.
        if params.max_completion_tokens is not None and params.max_tokens is None:
            converted["max_tokens"] = params.max_completion_tokens
        if params.stream and params.stream_options is not None:
            converted["stream_options"] = params.stream_options
        if params.response_format is not None:
            if is_structured_output_type(params.response_format):
                converted["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": params.response_format.__name__,
                        "schema": get_json_schema(params.response_format),
                    },
                }
            else:
                converted["response_format"] = params.response_format

        timeout = kwargs.pop("timeout", None)
        converted.update(kwargs)

        request_options = dict(converted.pop("request_options", None) or {})
        if timeout is not None:
            request_options.setdefault("timeout_in_seconds", timeout)
        # The API forwards OpenAI fields it does not document (user, logit_bias, ...) to the model,
        # but the SDK method raises TypeError on them, so they travel as extra body fields.
        extra_body_params = {key: converted.pop(key) for key in list(converted) if key not in _SDK_COMPLETION_PARAMS}
        if extra_body_params:
            request_options["additional_body_parameters"] = {
                **extra_body_params,
                **(request_options.get("additional_body_parameters") or {}),
            }
        if request_options:
            converted["request_options"] = request_options
        return converted

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        return _convert_response(response)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        return _convert_chunk(response)

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        msg = "Sarvam does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        msg = "Sarvam does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        return _convert_models_list(response)

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self._max_retries = kwargs.pop("max_retries", None)
        environment = SarvamAIEnvironment.PRODUCTION
        if api_base:
            # Chat and models calls only use `base`; the other two serve speech and dubbing.
            environment = SarvamAIEnvironment(
                base=api_base.rstrip("/"),
                creative=SarvamAIEnvironment.PRODUCTION.creative,
                production=SarvamAIEnvironment.PRODUCTION.production,
            )
        self.client = AsyncSarvamAI(api_subscription_key=api_key, environment=environment, **kwargs)

    async def _stream_completion_async(self, stream: AsyncIterator[Any]) -> AsyncIterator[ChatCompletionChunk]:
        try:
            async for chunk in stream:
                yield self._convert_completion_chunk_response(chunk)
        finally:
            await aclose_quietly(stream)

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        # Sarvam replays reasoning as `reasoning_content`, the field the Messages bridge already
        # emits, so only any_llm's `extra_content` side-channel has to go.
        messages = strip_extra_content(params.messages)
        completion_kwargs = self._convert_completion_params(params, **kwargs)
        request_options = self._request_options(completion_kwargs.pop("request_options", None))
        if request_options is not None:
            completion_kwargs["request_options"] = request_options

        if params.stream:
            stream = await self.client.chat.completions_v2(
                model=params.model_id,
                messages=messages,
                stream=True,
                **completion_kwargs,
            )
            return self._stream_completion_async(stream)

        response = await self.client.chat.completions_v2(
            model=params.model_id,
            messages=messages,
            **completion_kwargs,
        )
        return self._convert_completion_response(response)

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        request_options = self._request_options(kwargs.pop("request_options", None))
        response = await self.client.chat_completion.list_models_v2(request_options=request_options, **kwargs)
        return self._convert_list_models_response(response)
