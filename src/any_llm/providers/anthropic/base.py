from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.types.messages import (
    MessageContentBlock,
    MessageResponse,
    MessagesParams,
    MessageStreamEvent,
    MessageUsage,
)

MISSING_PACKAGES_ERROR = None
try:
    from anthropic import AsyncAnthropic

    from .utils import (
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
    from anthropic.types.model_info import ModelInfo as AnthropicModelInfo

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.model import Model


class BaseAnthropicProvider(AnyLLM, ABC):
    """
    Base provider for Anthropic-compatible services.

    This class provides a common foundation for providers that use Anthropic-compatible APIs.
    Subclasses need to override `_init_client()` for provider-specific client initialization.
    """

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = False
    SUPPORTS_BATCH = False

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
        """Native Anthropic Messages API pass-through.

        Avoids double-conversion (Anthropic→OpenAI→Anthropic) by calling the
        Anthropic SDK directly.
        """
        api_kwargs: dict[str, Any] = {
            "model": params.model,
            "messages": params.messages,
            "max_tokens": params.max_tokens,
        }
        if params.system is not None:
            api_kwargs["system"] = params.system
        if params.temperature is not None:
            api_kwargs["temperature"] = params.temperature
        if params.top_p is not None:
            api_kwargs["top_p"] = params.top_p
        if params.top_k is not None:
            api_kwargs["top_k"] = params.top_k
        if params.stop_sequences is not None:
            api_kwargs["stop_sequences"] = params.stop_sequences
        if params.tools is not None:
            api_kwargs["tools"] = params.tools
        if params.tool_choice is not None:
            api_kwargs["tool_choice"] = params.tool_choice
        if params.metadata is not None:
            api_kwargs["metadata"] = params.metadata
        if params.thinking is not None:
            api_kwargs["thinking"] = params.thinking
        api_kwargs.update(kwargs)

        if params.stream:
            return self._stream_messages_async(**api_kwargs)

        response: Message = await self.client.messages.create(**api_kwargs)
        return self._convert_native_message_to_response(response)

    async def _stream_messages_async(self, **kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        """Stream Anthropic Messages API and yield MessageStreamEvents."""
        from anthropic.types import (
            ContentBlockDeltaEvent,
            ContentBlockStartEvent,
            ContentBlockStopEvent,
            MessageStartEvent,
            MessageStopEvent,
        )

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if isinstance(event, MessageStartEvent):
                    msg = event.message
                    usage = MessageUsage(
                        input_tokens=msg.usage.input_tokens,
                        output_tokens=msg.usage.output_tokens,
                        cache_creation_input_tokens=getattr(msg.usage, "cache_creation_input_tokens", None),
                        cache_read_input_tokens=getattr(msg.usage, "cache_read_input_tokens", None),
                    )
                    resp = MessageResponse(
                        id=msg.id,
                        type="message",
                        role=msg.role,
                        content=[],
                        model=msg.model,
                        stop_reason=None,
                        usage=usage,
                    )
                    yield MessageStreamEvent(type="message_start", message=resp)

                elif isinstance(event, ContentBlockStartEvent):
                    block = self._convert_native_content_block(event.content_block)
                    yield MessageStreamEvent(
                        type="content_block_start",
                        index=event.index,
                        content_block=block,
                    )

                elif isinstance(event, ContentBlockDeltaEvent):
                    delta_dict: dict[str, Any] = {"type": event.delta.type}
                    if event.delta.type == "text_delta":
                        delta_dict["text"] = event.delta.text
                    elif event.delta.type == "input_json_delta":
                        delta_dict["partial_json"] = event.delta.partial_json
                    elif event.delta.type == "thinking_delta":
                        delta_dict["thinking"] = event.delta.thinking
                    yield MessageStreamEvent(
                        type="content_block_delta",
                        index=event.index,
                        delta=delta_dict,
                    )

                elif isinstance(event, ContentBlockStopEvent):
                    yield MessageStreamEvent(type="content_block_stop", index=event.index)

                elif isinstance(event, MessageStopEvent):
                    msg_obj = event.message
                    usage = MessageUsage(
                        input_tokens=msg_obj.usage.input_tokens,
                        output_tokens=msg_obj.usage.output_tokens,
                        cache_creation_input_tokens=getattr(msg_obj.usage, "cache_creation_input_tokens", None),
                        cache_read_input_tokens=getattr(msg_obj.usage, "cache_read_input_tokens", None),
                    )
                    yield MessageStreamEvent(
                        type="message_delta",
                        delta={"stop_reason": msg_obj.stop_reason},
                        usage=usage,
                    )
                    yield MessageStreamEvent(type="message_stop")

    @staticmethod
    def _convert_native_content_block(block: Any) -> MessageContentBlock:
        """Convert an Anthropic SDK content block to our MessageContentBlock."""
        if block.type == "text":
            return MessageContentBlock(type="text", text=block.text)
        if block.type == "tool_use":
            return MessageContentBlock(
                type="tool_use",
                id=block.id,
                name=block.name,
                input=block.input if isinstance(block.input, dict) else {},
            )
        if block.type == "thinking":
            return MessageContentBlock(type="thinking", thinking=block.thinking)
        return MessageContentBlock(type=block.type)

    @staticmethod
    def _convert_native_message_to_response(message: Message) -> MessageResponse:
        """Convert an Anthropic SDK Message to our MessageResponse."""
        content_blocks: list[MessageContentBlock] = []
        for block in message.content:
            if block.type == "text":
                content_blocks.append(MessageContentBlock(type="text", text=block.text))
            elif block.type == "tool_use":
                tool_input = block.input if isinstance(block.input, dict) else {}
                content_blocks.append(
                    MessageContentBlock(
                        type="tool_use",
                        id=block.id,
                        name=block.name,
                        input=tool_input,
                    )
                )
            elif block.type == "thinking":
                content_blocks.append(MessageContentBlock(type="thinking", thinking=block.thinking))

        usage = MessageUsage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            cache_creation_input_tokens=getattr(message.usage, "cache_creation_input_tokens", None),
            cache_read_input_tokens=getattr(message.usage, "cache_read_input_tokens", None),
        )

        return MessageResponse(
            id=message.id,
            type="message",
            role=message.role,
            content=content_blocks,
            model=message.model,
            stop_reason=message.stop_reason,
            usage=usage,
        )
