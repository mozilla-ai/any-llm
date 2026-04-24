from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import BatchNotCompleteError
from any_llm.logging import logger
from any_llm.types.batch import Batch, BatchRequestCounts, BatchResult, BatchResultError, BatchResultItem
from any_llm.types.messages import (
    ContentBlockStartEvent,
    MessageResponse,
    MessagesParams,
    MessageStreamEvent,
    ThinkingBlock,
)
from any_llm.types.request import RequestParams, RequestResponse, RequestStreamEvent
from any_llm.utils.request_output import RequestEventStream

MISSING_PACKAGES_ERROR = None
try:
    from anthropic import AsyncAnthropic

    from .utils import (
        _convert_request_input,
        _convert_request_response,
        _convert_request_tool_choice,
        _convert_request_tools,
        _convert_models_list,
        _convert_params,
        _convert_response,
        _create_openai_chunk_from_anthropic_chunk,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from anthropic import AsyncAnthropic, AsyncAnthropicVertex
    from anthropic.types import Message
    from anthropic.types.messages.message_batch import MessageBatch
    from anthropic.types.model_info import ModelInfo as AnthropicModelInfo

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model


_ANTHROPIC_TO_OPENAI_STATUS_MAP: dict[str, str] = {
    "in_progress": "in_progress",
    "canceling": "cancelling",
    "canceled": "cancelled",
    "expired": "expired",
}


def _convert_anthropic_batch_to_openai(batch: MessageBatch) -> Batch:
    """Convert an Anthropic MessageBatch to OpenAI Batch format."""
    status_str = batch.processing_status
    openai_status: str
    if status_str == "ended":
        openai_status = "completed"
    else:
        mapped_status = _ANTHROPIC_TO_OPENAI_STATUS_MAP.get(status_str)
        if mapped_status is None:
            logger.warning("Unknown Anthropic batch status: %s, defaulting to 'in_progress'", status_str)
            openai_status = "in_progress"
        else:
            openai_status = mapped_status

    request_counts = BatchRequestCounts(
        total=(
            batch.request_counts.processing
            + batch.request_counts.succeeded
            + batch.request_counts.errored
            + batch.request_counts.canceled
            + batch.request_counts.expired
        ),
        completed=batch.request_counts.succeeded,
        failed=batch.request_counts.errored + batch.request_counts.canceled + batch.request_counts.expired,
    )

    created_at = int(batch.created_at.timestamp()) if batch.created_at else 0

    return Batch(
        id=batch.id,
        object="batch",
        endpoint="/v1/chat/completions",
        status=cast("Any", openai_status),
        created_at=created_at,
        completion_window="24h",
        request_counts=request_counts,
        input_file_id="",
        output_file_id=None,
        error_file_id=None,
        metadata=None,
    )


class BaseAnthropicProvider(AnyLLM, ABC):
    """
    Base provider for Anthropic-compatible services.

    This class provides a common foundation for providers that use Anthropic-compatible APIs.
    Subclasses need to override `_init_client()` for provider-specific client initialization.
    """

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_REQUESTS = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = False
    SUPPORTS_BATCH = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncAnthropic | AsyncAnthropicVertex

    @abstractmethod
    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        msg = "Subclasses must implement this method"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Anthropic API."""
        return _convert_params(params, **kwargs)

    @staticmethod
    @override
    def _convert_completion_response(response: Message) -> ChatCompletion:
        """Convert Anthropic Message to OpenAI ChatCompletion format."""
        return _convert_response(response)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        """Convert Anthropic streaming chunk to OpenAI ChatCompletionChunk format."""
        model_id = kwargs.get("model_id", "unknown")
        return _create_openai_chunk_from_anthropic_chunk(response, model_id)

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        """Anthropic does not support embeddings."""
        msg = "Anthropic does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        """Anthropic does not support embeddings."""
        msg = "Anthropic does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_list_models_response(response: list[AnthropicModelInfo]) -> Sequence[Model]:
        """Convert Anthropic models list to OpenAI format."""
        return _convert_models_list(response)

    async def _stream_completion_async(self, **kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        """Handle streaming completion - extracted to avoid generator issues."""
        async with self.client.messages.stream(**kwargs) as anthropic_stream:
            async for event in anthropic_stream:
                yield self._convert_completion_chunk_response(event, model_id=kwargs.get("model", "unknown"))

    async def _stream_request_async(
        self, api_kwargs: dict[str, Any], params: RequestParams
    ) -> AsyncIterator[RequestStreamEvent]:
        response = await self.client.messages.create(**api_kwargs)
        resource = _convert_request_response(response, params=params)
        event_stream = RequestEventStream(resource)
        for event in event_stream.iter_preamble():
            yield event
        for output_index, item in enumerate(resource.output):
            for event in event_stream.iter_item(output_index, item):
                yield event
        for event in event_stream.iter_complete():
            yield event

    @override
    async def _acompletion(
        self,
        params: CompletionParams,
        **kwargs: Any,
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        kwargs["provider_name"] = self.PROVIDER_NAME
        converted_kwargs = self._convert_completion_params(params, **kwargs)

        if converted_kwargs.pop("stream", False):
            return self._stream_completion_async(**converted_kwargs)

        message = await self.client.messages.create(**converted_kwargs)

        return self._convert_completion_response(message)

    @override
    async def _amessages(
        self, params: MessagesParams, **kwargs: Any
    ) -> MessageResponse | AsyncIterator[MessageStreamEvent]:
        """Native Anthropic Messages API pass-through."""
        api_kwargs = params.model_dump(exclude_none=True)
        api_kwargs.pop("stream", None)
        api_kwargs.update(kwargs)

        if params.stream:
            return self._stream_messages_async(**api_kwargs)

        response: Message = await self.client.messages.create(**api_kwargs)
        return self._convert_native_message_to_response(response)

    @override
    async def _arequest(
        self, params: RequestParams, **kwargs: Any
    ) -> RequestResponse | AsyncIterator[RequestStreamEvent]:
        api_kwargs = params.model_dump(exclude_none=True)
        api_kwargs["messages"] = []
        system, messages = _convert_request_input(params.input, params.instructions)
        api_kwargs["messages"] = messages
        if system is not None:
            api_kwargs["system"] = system
        else:
            api_kwargs.pop("instructions", None)
        api_kwargs.pop("input", None)
        api_kwargs.pop("instructions", None)
        if api_kwargs.get("max_output_tokens") is not None:
            api_kwargs["max_tokens"] = api_kwargs.pop("max_output_tokens")
        elif "max_tokens" not in api_kwargs:
            api_kwargs["max_tokens"] = 1024
        if api_kwargs.get("tools") is not None:
            api_kwargs["tools"] = _convert_request_tools(api_kwargs["tools"])
        if api_kwargs.get("tool_choice") is not None:
            api_kwargs["tool_choice"] = _convert_request_tool_choice(api_kwargs["tool_choice"])
        if params.reasoning is not None:
            effort = params.reasoning.get("effort")
            budget_tokens = params.reasoning.get("budget_tokens")
            if budget_tokens is None:
                budget_tokens = 8192
                if effort == "high":
                    budget_tokens = 16384
                elif effort == "low":
                    budget_tokens = 4096
            api_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
        api_kwargs.pop("stream", None)
        if params.stream:
            return self._stream_request_async({**api_kwargs, **kwargs}, params)
        response = await self.client.messages.create(**api_kwargs, **kwargs)
        return _convert_request_response(response, params=params)

    async def _stream_messages_async(self, **kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        """Stream Anthropic Messages API and yield SDK event types directly."""
        from anthropic.types import (
            RawContentBlockDeltaEvent,
            RawContentBlockStartEvent,
            RawContentBlockStopEvent,
            RawMessageDeltaEvent,
            RawMessageStartEvent,
            RawMessageStopEvent,
        )
        from anthropic.types import (
            ThinkingBlock as AnthropicThinkingBlock,
        )

        raw_types = (
            RawMessageStartEvent,
            RawMessageDeltaEvent,
            RawMessageStopEvent,
            RawContentBlockStartEvent,
            RawContentBlockDeltaEvent,
            RawContentBlockStopEvent,
        )

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if not isinstance(event, raw_types):
                    continue
                if isinstance(event, RawContentBlockStartEvent) and isinstance(
                    event.content_block, AnthropicThinkingBlock
                ):
                    yield ContentBlockStartEvent(
                        type="content_block_start",
                        index=event.index,
                        content_block=ThinkingBlock(
                            type="thinking",
                            thinking=event.content_block.thinking,
                            signature=event.content_block.signature,
                        ),
                    )
                else:
                    yield event

    @staticmethod
    def _convert_native_message_to_response(message: Message) -> MessageResponse:
        """Convert an Anthropic SDK Message to our MessageResponse."""
        return MessageResponse.model_validate(message, from_attributes=True)

    @override
    async def _acreate_batch(
        self,
        input_file_path: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Batch:
        """Create a batch job using the Anthropic Messages Batches API."""
        file_path = Path(input_file_path)
        file_content = await asyncio.to_thread(file_path.read_text)

        requests = []
        for line in file_content.strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            body = entry.get("body", {})
            params: dict[str, Any] = {
                "model": body.get("model", ""),
                "max_tokens": body.get("max_tokens", 1024),
                "messages": body.get("messages", []),
            }
            if "temperature" in body:
                params["temperature"] = body["temperature"]
            if "top_p" in body:
                params["top_p"] = body["top_p"]
            if "system" in body:
                params["system"] = body["system"]
            requests.append(
                {
                    "custom_id": entry["custom_id"],
                    "params": params,
                }
            )

        result = await self.client.messages.batches.create(requests=requests)  # type: ignore[arg-type]
        return _convert_anthropic_batch_to_openai(result)

    @override
    async def _aretrieve_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Retrieve a batch job using the Anthropic Messages Batches API."""
        result = await self.client.messages.batches.retrieve(batch_id)
        return _convert_anthropic_batch_to_openai(result)

    @override
    async def _acancel_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Cancel a batch job using the Anthropic Messages Batches API."""
        result = await self.client.messages.batches.cancel(batch_id)
        return _convert_anthropic_batch_to_openai(result)

    @override
    async def _alist_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Sequence[Batch]:
        """List batch jobs using the Anthropic Messages Batches API."""
        list_kwargs: dict[str, Any] = {}
        if after:
            list_kwargs["after_id"] = after
        if limit:
            list_kwargs["limit"] = limit
        result = await self.client.messages.batches.list(**list_kwargs)
        return [_convert_anthropic_batch_to_openai(b) for b in result.data]

    @override
    async def _aretrieve_batch_results(self, batch_id: str, **kwargs: Any) -> BatchResult:
        """Retrieve the results of a completed batch job using the Anthropic Messages Batches API."""
        batch = await self.client.messages.batches.retrieve(batch_id)
        if batch.processing_status != "ended":
            openai_batch = _convert_anthropic_batch_to_openai(batch)
            raise BatchNotCompleteError(
                batch_id=batch_id,
                status=openai_batch.status or "unknown",
                provider_name=self.PROVIDER_NAME,
            )

        results: list[BatchResultItem] = []
        async for entry in await self.client.messages.batches.results(batch_id):
            item = BatchResultItem(custom_id=entry.custom_id)
            if entry.result.type == "succeeded":
                item.result = _convert_response(entry.result.message)
            elif entry.result.type == "errored":
                err = entry.result.error
                item.error = BatchResultError(
                    code=err.error.type if err and err.error else "unknown",
                    message=err.error.message if err and err.error else "Unknown error",
                )
            else:
                item.error = BatchResultError(code=entry.result.type, message=f"Request {entry.result.type}")
            results.append(item)
        return BatchResult(results=results)
