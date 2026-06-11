from __future__ import annotations

import importlib
import json
import os
from typing import TYPE_CHECKING, Any, TypedDict, cast

from pydantic import BaseModel
from typing_extensions import override

from any_llm.exceptions import BatchNotCompleteError, InvalidRequestError
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.openai.utils import _convert_moderation_response_from_openai
from any_llm.types.batch import Batch, BatchResult, BatchResultError, BatchResultItem
from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    CompletionParams,
    CreateEmbeddingResponse,
    ParsedChatCompletion,
)
from any_llm.types.messages import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
)
from any_llm.types.model import Model
from any_llm.types.rerank import RerankResponse
from any_llm.utils.structured_output import (
    build_responses_text_format,
    get_json_schema,
    is_structured_output_type,
    parse_json_content,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from openresponses_types import ResponseResource

    from any_llm.types.audio import AudioSpeechParams, AudioTranscriptionParams, Transcription
    from any_llm.types.image import ImageGenerationParams, ImagesResponse
    from any_llm.types.messages import MessagesParams, MessageStreamEvent
    from any_llm.types.moderation import ModerationResponse
    from any_llm.types.responses import Response, ResponsesParams, ResponseStreamEvent


class CreateBatchParams(TypedDict, total=False):
    model: str
    requests: list[dict[str, Any]]
    completion_window: str
    metadata: dict[str, str]


class ListBatchesOptions(TypedDict, total=False):
    after: str
    limit: int


_MISSING_PACKAGES_ERROR: ImportError | None = None
_OTARI_BATCH_NOT_COMPLETE_ERROR: Any = None
try:
    AsyncOtariClient = importlib.import_module("otari").AsyncOtariClient
    _OTARI_BATCH_NOT_COMPLETE_ERROR = importlib.import_module("otari.errors").BatchNotCompleteError
except ImportError as e:  # pragma: no cover - exercised through MISSING_PACKAGES_ERROR checks
    _MISSING_PACKAGES_ERROR = e


# Canonical platform-token env var exposed by otari 0.1.0; the others are kept as
# legacy aliases for backwards compatibility.
OTARI_AI_TOKEN_ENV = "OTARI_AI_TOKEN"  # noqa: S105
OTARI_PLATFORM_TOKEN_ENV = "OTARI_PLATFORM_TOKEN"  # noqa: S105
GATEWAY_PLATFORM_TOKEN_ENV = "GATEWAY_PLATFORM_TOKEN"  # noqa: S105
OTARI_HEADER_NAME = "Otari-Key"
LEGACY_GATEWAY_HEADER_NAME = "AnyLLM-Key"

LEGACY_GATEWAY_API_KEY_ENV = "GATEWAY_API_KEY"
LEGACY_GATEWAY_API_BASE_ENV = "GATEWAY_API_BASE"


def _as_plain_dict(response: Any) -> Any:
    """Normalize an otari-native pydantic response into a plain dict.

    otari 0.1.0's async client returns its own typed pydantic models rather than
    OpenAI types. Dumping them to a dict lets any-llm's own response types validate
    the payload (extra keys like ``additional_properties`` are ignored/allowed).
    Non-model inputs (e.g. dicts in unit tests) are returned unchanged.
    """
    if isinstance(response, BaseModel):
        return response.model_dump()
    return response


def _parse_jsonl_to_requests(file_path: str) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    with open(file_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            entry = json.loads(stripped)
            requests.append(
                {
                    "custom_id": entry["custom_id"],
                    "body": entry.get("body", {}),
                }
            )
    return requests


def _extract_model_from_requests(requests: list[dict[str, Any]]) -> str | None:
    if requests and requests[0].get("body"):
        model: Any = requests[0]["body"].get("model")
        return str(model) if model is not None else None
    return None


# otari's /messages stream yields raw Anthropic SSE event dicts (the SDK has no
# single typed model for them). Map each by its ``type`` field to the matching
# any-llm event model; unknown types (e.g. ``ping``) are skipped.
_MESSAGE_STREAM_EVENT_TYPES: dict[str, type[BaseModel]] = {
    "message_start": MessageStartEvent,
    "message_delta": MessageDeltaEvent,
    "message_stop": MessageStopEvent,
    "content_block_start": ContentBlockStartEvent,
    "content_block_delta": ContentBlockDeltaEvent,
    "content_block_stop": ContentBlockStopEvent,
}


def _message_stream_event_from_dict(event: dict[str, Any]) -> MessageStreamEvent | None:
    """Validate a raw otari message SSE event dict into a typed MessageStreamEvent.

    Returns ``None`` for event types any-llm does not surface (e.g. ``ping``).
    """
    event_type = event.get("type")
    model = _MESSAGE_STREAM_EVENT_TYPES.get(event_type) if isinstance(event_type, str) else None
    if model is None:
        return None
    return cast("MessageStreamEvent", model.model_validate(event))


class OtariProvider(BaseOpenAIProvider):
    ENV_API_KEY_NAME = "OTARI_API_KEY"
    ENV_API_BASE_NAME = "OTARI_API_BASE"
    PROVIDER_NAME = "otari"
    PROVIDER_DOCUMENTATION_URL = "https://mozilla-ai.github.io/otari/"
    MISSING_PACKAGES_ERROR = _MISSING_PACKAGES_ERROR

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_MODERATION = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = True
    # otari 0.1.0's public async client dropped the embedded OpenAI passthrough and
    # does not expose image generation or audio (speech/transcription). Tracked for
    # upstream re-enablement; see the otari issue linked in the any-llm PR.
    SUPPORTS_IMAGE_GENERATION = False
    SUPPORTS_AUDIO_TRANSCRIPTION = False
    SUPPORTS_AUDIO_SPEECH = False
    SUPPORTS_RERANK = True

    otari_client: Any

    @classmethod
    def _resolve_env_api_base(cls) -> str | None:
        return os.getenv(cls.ENV_API_BASE_NAME) or os.getenv(LEGACY_GATEWAY_API_BASE_ENV)

    @classmethod
    def _resolve_env_api_key(cls) -> str | None:
        return os.getenv(cls.ENV_API_KEY_NAME) or os.getenv(LEGACY_GATEWAY_API_KEY_ENV)

    @staticmethod
    def _resolve_platform_token() -> str | None:
        return (
            os.getenv(OTARI_AI_TOKEN_ENV)
            or os.getenv(OTARI_PLATFORM_TOKEN_ENV)
            or os.getenv(GATEWAY_PLATFORM_TOKEN_ENV)
        )

    @override
    def _resolve_api_base(self, api_base: str | None = None) -> str | None:
        if api_base:
            return api_base
        return self._resolve_env_api_base()

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str:
        return api_key or self._resolve_env_api_key() or "no-key-required"

    @staticmethod
    def _normalize_placeholder_api_key(api_key: str | None) -> str | None:
        return None if api_key == "no-key-required" else api_key

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        platform_mode = kwargs.pop("platform_mode", None)
        default_headers = kwargs.pop("default_headers", None)

        resolved_api_base = self._resolve_api_base(api_base)
        normalized_api_key = self._normalize_placeholder_api_key(api_key)
        resolved_api_key = normalized_api_key or self._resolve_env_api_key()
        resolved_platform_token = self._resolve_platform_token()

        client_kwargs: dict[str, Any] = {"default_headers": default_headers}
        if resolved_api_base:
            client_kwargs["api_base"] = resolved_api_base

        # Track whether the client will authenticate with a platform token. otari
        # defaults api_base to the hosted gateway in that case, so api_base is optional.
        using_platform_token = False
        if platform_mode is True:
            token = normalized_api_key or resolved_platform_token
            if not token:
                msg = (
                    "Platform mode requires a user token (pass api_key or set "
                    f"{OTARI_AI_TOKEN_ENV}/{OTARI_PLATFORM_TOKEN_ENV}/{GATEWAY_PLATFORM_TOKEN_ENV})"
                )
                raise ValueError(msg)
            client_kwargs["platform_token"] = token
            using_platform_token = True
        elif platform_mode is False:
            client_kwargs["api_key"] = resolved_api_key
        elif normalized_api_key:
            client_kwargs["api_key"] = normalized_api_key
        elif resolved_platform_token:
            client_kwargs["platform_token"] = resolved_platform_token
            using_platform_token = True
        elif resolved_api_key:
            client_kwargs["api_key"] = resolved_api_key

        if not resolved_api_base and not using_platform_token:
            msg = (
                "For any-llm otari, api_base is required unless a platform token is provided "
                f"(set via parameter or {self.ENV_API_BASE_NAME}/{LEGACY_GATEWAY_API_BASE_ENV} env vars). "
                "With a platform token, otari defaults to its hosted gateway."
            )
            raise ValueError(msg)

        self.otari_client = AsyncOtariClient(**client_kwargs)
        # otari 0.1.0's async client has no embedded OpenAI passthrough; every base
        # client-based method is overridden below, so self.client is never used for I/O.
        self.client = self.otari_client
        self.platform_mode = self.otari_client.platform_mode

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert completion params for the otari gateway.

        The base OpenAI layer remaps ``max_tokens`` to ``max_completion_tokens`` to follow
        the current OpenAI spec, but the otari gateway accepts only ``max_tokens`` and errors
        on ``max_completion_tokens``. Remap it back so the token limit reaches the upstream
        provider instead of producing an upstream error.

        The base also leaves a Pydantic ``response_format`` as the model class (the OpenAI SDK
        consumes it via ``parse()``); otari has no ``parse()`` helper, so send it as a
        json_schema dict instead. The parsed object is reconstructed in ``_acompletion``.
        """
        converted = BaseOpenAIProvider._convert_completion_params(params, **kwargs)
        if "max_completion_tokens" in converted:
            converted["max_tokens"] = converted.pop("max_completion_tokens")
        response_format = converted.get("response_format")
        if is_structured_output_type(response_format):
            converted["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": response_format.__name__, "schema": get_json_schema(response_format)},
            }
        return converted

    @staticmethod
    def _to_parsed_completion(completion: ChatCompletion, response_format: type) -> ParsedChatCompletion[Any]:
        """Wrap a completion as a ParsedChatCompletion, parsing each message's JSON content."""
        parsed: ParsedChatCompletion[Any] = ParsedChatCompletion.model_validate(completion, from_attributes=True)
        for choice in parsed.choices:
            if choice.message.content is not None:
                choice.message.parsed = parse_json_content(response_format, choice.message.content)
        return parsed

    @override
    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        # Capture the original response_format before _convert_completion_params rewrites it,
        # so the JSON content can be parsed back into the requested type.
        response_format = params.response_format
        wants_structured = is_structured_output_type(response_format)
        completion_kwargs = self._convert_completion_params(params, **kwargs)
        if wants_structured:
            if params.stream:
                msg = "stream is not supported for response_format"
                raise ValueError(msg)
            completion_kwargs.pop("stream", None)

        response = await self.otari_client.completion(
            model=params.model_id,
            messages=params.messages,
            **completion_kwargs,
        )
        # otari returns its own typed ChatCompletion (non-stream) or an async iterator
        # of typed chunks (stream); normalize both to any-llm types.
        if hasattr(response, "__aiter__"):

            async def _chunk_iterator() -> AsyncIterator[ChatCompletionChunk]:
                async for chunk in response:
                    yield self._convert_completion_chunk_response(_as_plain_dict(chunk))

            return _chunk_iterator()
        completion = self._convert_completion_response(_as_plain_dict(response))
        # Re-check inline (rather than reusing ``wants_structured``) so the TypeGuard narrows
        # response_format to ``type`` for _to_parsed_completion.
        if is_structured_output_type(response_format):
            return self._to_parsed_completion(completion, response_format)
        return completion

    @override
    async def _amessages(
        self, params: MessagesParams, **kwargs: Any
    ) -> MessageResponse | AsyncIterator[MessageStreamEvent]:
        """Native Anthropic Messages API pass-through via otari's /messages endpoint.

        The base implementation converts Messages to Chat Completions, which silently
        drops Anthropic-only features (``cache_control`` on system blocks, ``thinking``
        config). otari's gateway serves /messages natively, so delegate to the otari
        SDK's ``message()`` to preserve them.
        """
        api_kwargs = params.model_dump(exclude_none=True)
        api_kwargs.update(kwargs)
        api_kwargs.pop("stream", None)

        if params.stream:
            return self._stream_messages_async(**api_kwargs)

        response = await self.otari_client.message(**api_kwargs)
        return MessageResponse.model_validate(_as_plain_dict(response))

    async def _stream_messages_async(self, **api_kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        """Stream otari's /messages endpoint, yielding typed any-llm event models."""
        stream = await self.otari_client.message(stream=True, **api_kwargs)
        async for event in stream:
            converted = _message_stream_event_from_dict(_as_plain_dict(event))
            if converted is not None:
                yield converted

    @override
    async def _aresponses(
        self, params: ResponsesParams, **kwargs: Any
    ) -> ResponseResource | Response | AsyncIterator[ResponseStreamEvent]:
        response_format = params.response_format
        response_kwargs = params.model_dump(exclude_none=True, exclude={"model", "input", "response_format"})
        # Otari has no parse() helper: request schema-conformant JSON via text.format and let the
        # base layer parse the result into a ParsedResponse.
        if is_structured_output_type(response_format):
            response_kwargs["text"] = build_responses_text_format(response_format)
        elif isinstance(response_format, dict):
            response_kwargs["text"] = {"format": response_format}
        response_kwargs.update(kwargs)
        result = await self.otari_client.response(model=params.model, input=params.input, **response_kwargs)
        return cast("ResponseResource | Response | AsyncIterator[ResponseStreamEvent]", result)

    @override
    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        result = await self.otari_client.embedding(model=model, input=inputs, **kwargs)
        return self._convert_embedding_response(_as_plain_dict(result))

    @override
    async def _aimage_generation(self, params: ImageGenerationParams, **kwargs: Any) -> ImagesResponse:
        msg = (
            "otari 0.1.0 does not expose image generation in its public async client. "
            "Re-enable once otari adds it upstream."
        )
        raise NotImplementedError(msg)

    @override
    async def _atranscription(self, params: AudioTranscriptionParams, **kwargs: Any) -> Transcription:
        msg = (
            "otari 0.1.0 does not expose audio transcription in its public async client. "
            "Re-enable once otari adds it upstream."
        )
        raise NotImplementedError(msg)

    @override
    async def _aspeech(self, params: AudioSpeechParams, **kwargs: Any) -> bytes:
        msg = (
            "otari 0.1.0 does not expose audio speech in its public async client. "
            "Re-enable once otari adds it upstream."
        )
        raise NotImplementedError(msg)

    @override
    async def _amoderation(
        self,
        model: str,
        input: str | list[str] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModerationResponse:
        include_raw = kwargs.pop("include_raw", False)
        model_name = model or "omni-moderation-latest"
        raw = await self.otari_client.moderation(
            model=model_name,
            input=cast("Any", input),
            **kwargs,
        )
        return _convert_moderation_response_from_openai(raw, include_raw=include_raw)

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        models = await self.otari_client.list_models()
        return [model if isinstance(model, Model) else Model.model_validate(_as_plain_dict(model)) for model in models]

    @override
    async def _acreate_batch(
        self,
        input_file_path: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Batch:
        if endpoint != "/v1/chat/completions":
            msg = "otari batch currently supports only /v1/chat/completions endpoint"
            raise InvalidRequestError(message=msg, provider_name=self.PROVIDER_NAME)

        requests = _parse_jsonl_to_requests(input_file_path)
        model = kwargs.pop("model", None) or _extract_model_from_requests(requests)
        payload: CreateBatchParams = {
            "requests": requests,
            "completion_window": completion_window,
        }
        if model is not None:
            payload["model"] = model
        if metadata:
            payload["metadata"] = metadata
        data = await self.otari_client.create_batch(payload)
        return Batch(**{k: v for k, v in data.items() if k != "provider"})

    @override
    async def _aretrieve_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        provider_name = kwargs.pop("provider_name", None)
        if not provider_name:
            msg = "provider_name is required for Otari batch operations"
            raise InvalidRequestError(message=msg, provider_name=self.PROVIDER_NAME)
        data = await self.otari_client.retrieve_batch(batch_id=batch_id, provider=provider_name)
        return Batch(**data)

    @override
    async def _acancel_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        provider_name = kwargs.pop("provider_name", None)
        if not provider_name:
            msg = "provider_name is required for Otari batch operations"
            raise InvalidRequestError(message=msg, provider_name=self.PROVIDER_NAME)
        data = await self.otari_client.cancel_batch(batch_id=batch_id, provider=provider_name)
        return Batch(**data)

    @override
    async def _alist_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Sequence[Batch]:
        provider_name = kwargs.pop("provider_name", None)
        if not provider_name:
            msg = "provider_name is required for Otari batch operations"
            raise InvalidRequestError(message=msg, provider_name=self.PROVIDER_NAME)
        options: ListBatchesOptions | None = None
        if after or limit is not None:
            options = {}
            if after:
                options["after"] = after
            if limit is not None:
                options["limit"] = limit

        data = await self.otari_client.list_batches(provider=provider_name, options=options)
        return [Batch(**batch) for batch in data]

    @override
    async def _aretrieve_batch_results(self, batch_id: str, **kwargs: Any) -> BatchResult:
        provider_name = kwargs.pop("provider_name", None)
        if not provider_name:
            msg = "provider_name is required for Otari batch operations"
            raise InvalidRequestError(message=msg, provider_name=self.PROVIDER_NAME)

        try:
            payload = cast(
                "Any", await self.otari_client.retrieve_batch_results(batch_id=batch_id, provider=provider_name)
            )
        except Exception as exc:
            if _OTARI_BATCH_NOT_COMPLETE_ERROR is not None and isinstance(exc, _OTARI_BATCH_NOT_COMPLETE_ERROR):
                raise BatchNotCompleteError(
                    batch_id=batch_id, status="unknown", provider_name=self.PROVIDER_NAME
                ) from exc
            raise

        raw_results: list[Any]
        if isinstance(payload, dict):
            raw_results = cast("list[Any]", payload.get("results", []))
        else:
            raw_results = cast("list[Any]", payload.results)

        results: list[BatchResultItem] = []
        for item in raw_results:
            if isinstance(item, dict):
                custom_id = str(item.get("custom_id", ""))
                raw_result = item.get("result")
                raw_error = item.get("error")
            else:
                custom_id = str(item.custom_id)
                raw_result = item.result
                raw_error = item.error

            parsed_error: BatchResultError | None = None
            if raw_error:
                if isinstance(raw_error, dict):
                    parsed_error = BatchResultError(
                        code=str(raw_error.get("code", "unknown")),
                        message=str(raw_error.get("message", "Unknown batch error")),
                    )
                else:
                    parsed_error = BatchResultError(
                        code=str(getattr(raw_error, "code", "unknown")),
                        message=str(getattr(raw_error, "message", "Unknown batch error")),
                    )

            results.append(
                BatchResultItem(
                    custom_id=custom_id,
                    result=ChatCompletion.model_validate(raw_result) if raw_result else None,
                    error=parsed_error,
                )
            )
        return BatchResult(results=results)

    @override
    async def _arerank(
        self,
        model: str,
        query: str,
        documents: list[str],
        **kwargs: Any,
    ) -> RerankResponse:
        result = await self.otari_client.rerank(model=model, query=query, documents=documents, **kwargs)
        return RerankResponse.model_validate(_as_plain_dict(result))
