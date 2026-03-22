from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.types.messages import (
    ContentBlockStartEvent,
    MessageResponse,
    MessagesParams,
    MessageStreamEvent,
    ThinkingBlock,
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
    from anthropic.types import ContentBlock as AnthropicContentBlock
    from anthropic.types import Message
    from anthropic.types.model_info import ModelInfo as AnthropicModelInfo

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse
    from any_llm.types.messages import ContentBlock
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
        """Native Anthropic Messages API pass-through."""
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
        if params.cache_control is not None:
            api_kwargs["cache_control"] = params.cache_control
        api_kwargs.update(kwargs)

        if params.stream:
            return self._stream_messages_async(**api_kwargs)

        response: Message = await self.client.messages.create(**api_kwargs)
        return self._convert_native_message_to_response(response)

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
    def _convert_native_content_block(block: AnthropicContentBlock) -> ContentBlock:
        """Convert an Anthropic SDK content block, wrapping ThinkingBlock to make signature optional."""
        from anthropic.types import ThinkingBlock as AnthropicThinkingBlock

        if isinstance(block, AnthropicThinkingBlock):
            return ThinkingBlock(type="thinking", thinking=block.thinking, signature=block.signature)
        return block  # type: ignore[return-value]

    @classmethod
    def _convert_native_message_to_response(cls, message: Message) -> MessageResponse:
        """Convert an Anthropic SDK Message to our MessageResponse."""
        content_blocks = [cls._convert_native_content_block(block) for block in message.content]

        return MessageResponse(
            id=message.id,
            type="message",
            role=message.role,
            content=content_blocks,
            model=message.model,
            stop_reason=message.stop_reason,
            usage=message.usage,
        )
