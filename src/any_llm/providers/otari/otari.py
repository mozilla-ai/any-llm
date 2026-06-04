from __future__ import annotations

import importlib
import json
import os
from typing import TYPE_CHECKING, Any, TypedDict, cast

from typing_extensions import override

from any_llm.exceptions import BatchNotCompleteError, InvalidRequestError
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.openai.utils import _convert_moderation_response_from_openai
from any_llm.types.batch import Batch, BatchResult, BatchResultError, BatchResultItem
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
from any_llm.types.messages import MessageResponse, MessagesParams, MessageStreamEvent
from any_llm.types.model import Model
from any_llm.types.rerank import RerankResponse
from any_llm.utils.structured_output import build_responses_text_format, is_structured_output_type

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from httpx import Response as _HttpxResponse
    from openresponses_types import ResponseResource

    from any_llm.types.audio import AudioSpeechParams, AudioTranscriptionParams, Transcription
    from any_llm.types.image import ImageGenerationParams, ImagesResponse
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
    OtariClient = importlib.import_module("otari").OtariClient
    _OTARI_BATCH_NOT_COMPLETE_ERROR = importlib.import_module("otari.errors").BatchNotCompleteError
except ImportError as e:  # pragma: no cover - exercised through MISSING_PACKAGES_ERROR checks
    _MISSING_PACKAGES_ERROR = e


OTARI_PLATFORM_TOKEN_ENV = "OTARI_PLATFORM_TOKEN"  # noqa: S105
GATEWAY_PLATFORM_TOKEN_ENV = "GATEWAY_PLATFORM_TOKEN"  # noqa: S105
OTARI_HEADER_NAME = "Otari-Key"
LEGACY_GATEWAY_HEADER_NAME = "AnyLLM-Key"

LEGACY_GATEWAY_API_KEY_ENV = "GATEWAY_API_KEY"
LEGACY_GATEWAY_API_BASE_ENV = "GATEWAY_API_BASE"


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
    SUPPORTS_IMAGE_GENERATION = True
    SUPPORTS_AUDIO_TRANSCRIPTION = True
    SUPPORTS_AUDIO_SPEECH = True
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
        return os.getenv(OTARI_PLATFORM_TOKEN_ENV) or os.getenv(GATEWAY_PLATFORM_TOKEN_ENV)

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
        resolved_api_base = self._resolve_api_base(api_base)
        if not resolved_api_base:
            msg = (
                "For any-llm otari, api_base is required "
                f"(set via parameter or {self.ENV_API_BASE_NAME}/{LEGACY_GATEWAY_API_BASE_ENV} env vars)"
            )
            raise ValueError(msg)

        platform_mode = kwargs.pop("platform_mode", None)
        default_headers = kwargs.pop("default_headers", None)

        normalized_api_key = self._normalize_placeholder_api_key(api_key)
        resolved_api_key = normalized_api_key or self._resolve_env_api_key()
        resolved_platform_token = self._resolve_platform_token()

        client_kwargs: dict[str, Any] = {
            "api_base": resolved_api_base,
            "default_headers": default_headers,
            "openai_options": kwargs or None,
        }

        if platform_mode is True:
            token = normalized_api_key or resolved_platform_token
            if not token:
                msg = (
                    "Platform mode requires a user token "
                    f"(pass api_key or set {OTARI_PLATFORM_TOKEN_ENV}/{GATEWAY_PLATFORM_TOKEN_ENV})"
                )
                raise ValueError(msg)
            client_kwargs["platform_token"] = token
        elif platform_mode is False:
            client_kwargs["api_key"] = resolved_api_key
        else:
            if normalized_api_key:
                client_kwargs["api_key"] = normalized_api_key
            elif resolved_platform_token:
                client_kwargs["platform_token"] = resolved_platform_token
            elif resolved_api_key:
                client_kwargs["api_key"] = resolved_api_key

        self.otari_client = OtariClient(**client_kwargs)
        self.client = self.otari_client.openai
        self.platform_mode = self.otari_client.platform_mode

    @override
    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        completion_kwargs = self._convert_completion_params(params, **kwargs)
        response = await self.otari_client.completion(
            model=params.model_id,
            messages=params.messages,
            **completion_kwargs,
        )
        return self._convert_completion_response_async(response)

    @override
    async def _amessages(
        self, params: MessagesParams, **kwargs: Any
    ) -> MessageResponse | AsyncIterator[MessageStreamEvent]:
        """Send Anthropic-format messages directly to otari's /v1/messages endpoint.

        This preserves cache_control, thinking, and system prompt structure that
        would be lost during Anthropic-to-OpenAI conversion.
        """
        body = params.model_dump(exclude_none=True)
        body.pop("stream", None)
        body.update(kwargs)

        url = f"{self.otari_client._base_url}/messages"
        headers = {
            "Content-Type": "application/json",
            **self.otari_client._auth_headers,
        }

        if params.stream:
            body["stream"] = True
            response = await self.otari_client._http.post(url, headers=headers, json=body)
            response.raise_for_status()
            return self._stream_anthropic_events(response)

        response = await self.otari_client._http.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()
        return MessageResponse.model_validate(data)

    async def _stream_anthropic_events(self, response: _HttpxResponse) -> AsyncIterator[MessageStreamEvent]:
        """Parse SSE events from an Anthropic Messages API streaming response."""
        from anthropic.types import (
            RawContentBlockDeltaEvent,
            RawContentBlockStartEvent,
            RawContentBlockStopEvent,
            RawMessageDeltaEvent,
            RawMessageStartEvent,
            RawMessageStopEvent,
        )

        buf = ""
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                event_type = payload.get("type")
                if event_type == "message_start":
                    yield RawMessageStartEvent.model_validate(payload)
                elif event_type == "content_block_start":
                    yield RawContentBlockStartEvent.model_validate(payload)
                elif event_type == "content_block_delta":
                    yield RawContentBlockDeltaEvent.model_validate(payload)
                elif event_type == "content_block_stop":
                    yield RawContentBlockStopEvent.model_validate(payload)
                elif event_type == "message_delta":
                    yield RawMessageDeltaEvent.model_validate(payload)
                elif event_type == "message_stop":
                    yield RawMessageStopEvent.model_validate(payload)
            elif line.startswith("data:"):
                # Handle continuation lines (data: without space)
                buf += line[5:]
            elif not line.strip() and buf:
                # Empty line signals end of event
                payload = json.loads(buf)
                event_type = payload.get("type")
                if event_type == "message_start":
                    yield RawMessageStartEvent.model_validate(payload)
                elif event_type == "content_block_start":
                    yield RawContentBlockStartEvent.model_validate(payload)
                elif event_type == "content_block_delta":
                    yield RawContentBlockDeltaEvent.model_validate(payload)
                elif event_type == "content_block_stop":
                    yield RawContentBlockStopEvent.model_validate(payload)
                elif event_type == "message_delta":
                    yield RawMessageDeltaEvent.model_validate(payload)
                elif event_type == "message_stop":
                    yield RawMessageStopEvent.model_validate(payload)
                buf = ""

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
        return self._convert_embedding_response(result)

    @override
    async def _aimage_generation(self, params: ImageGenerationParams, **kwargs: Any) -> ImagesResponse:
        api_kwargs = params.to_api_kwargs()
        api_kwargs.update(kwargs)
        return await self.otari_client.openai.images.generate(model=params.model_id, **api_kwargs)  # type: ignore[no-any-return]

    @override
    async def _atranscription(self, params: AudioTranscriptionParams, **kwargs: Any) -> Transcription:
        api_kwargs = params.to_api_kwargs()
        api_kwargs.update(kwargs)
        return await self.otari_client.openai.audio.transcriptions.create(  # type: ignore[no-any-return]
            model=params.model_id,
            file=params.file,
            **api_kwargs,
        )

    @override
    async def _aspeech(self, params: AudioSpeechParams, **kwargs: Any) -> bytes:
        api_kwargs = params.to_api_kwargs()
        api_kwargs.update(kwargs)
        response = await self.otari_client.openai.audio.speech.create(
            model=params.model_id,
            input=params.input,
            voice=params.voice,
            **api_kwargs,
        )
        return cast("bytes", response.content)

    @override
    async def _amoderation(
        self,
        model: str,
        input: str | list[str] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModerationResponse:
        include_raw = kwargs.pop("include_raw", False)
        model_name = model or "omni-moderation-latest"
        raw = await self.otari_client.openai.moderations.create(
            model=model_name,
            input=cast("Any", input),
            **kwargs,
        )
        return _convert_moderation_response_from_openai(raw, include_raw=include_raw)

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        models = await self.otari_client.list_models()
        return [Model.model_validate(model) if not isinstance(model, Model) else model for model in models]

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
        if not hasattr(self.otari_client, "rerank"):
            msg = "Installed otari SDK does not expose rerank(). Upgrade otari to use rerank support."
            raise NotImplementedError(msg)
        rerank_fn = cast("Any", self.otari_client).rerank
        result = await rerank_fn(model=model, query=query, documents=documents, **kwargs)
        return RerankResponse.model_validate(result)
