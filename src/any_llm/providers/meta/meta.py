from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from typing_extensions import override

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.messages import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
)
from any_llm.utils.structured_output import is_structured_output_type

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from anthropic.types.beta.parsed_beta_message import ParsedBetaMessage
    from anthropic.types.parsed_message import ParsedMessage

    from any_llm.types.messages import MessagesParams, MessageStreamEvent

# Meta's /v1/messages stream yields typed anthropic SDK events (unlike gateways that hand
# back raw dicts), so events are validated straight into any-llm's typed models by `type`.
_MESSAGE_STREAM_EVENT_TYPES: dict[str, type[Any]] = {
    "message_start": MessageStartEvent,
    "message_delta": MessageDeltaEvent,
    "message_stop": MessageStopEvent,
    "content_block_start": ContentBlockStartEvent,
    "content_block_delta": ContentBlockDeltaEvent,
    "content_block_stop": ContentBlockStopEvent,
}


def _derive_anthropic_base(openai_base: str) -> str:
    """Derive the Anthropic SDK base URL from the OpenAI SDK one.

    Meta's OpenAI-compatible endpoints live under `.../v1` (the OpenAI SDK appends
    `/chat/completions`, `/responses`, etc. itself); the Anthropic SDK appends
    `/v1/messages` itself, so it needs the bare host instead.
    """
    return openai_base.removesuffix("/v1")


class MetaProvider(BaseOpenAIProvider):
    """Meta Model API provider.

    Chat Completions and Responses are OpenAI-SDK compatible and served through the
    inherited `BaseOpenAIProvider` machinery. The Messages API is served natively through
    the Anthropic SDK, pointed at Meta's Anthropic-compatible endpoint, rather than through
    any-llm's default Messages<->Completions bridge, so that `thinking` blocks, native
    `tool_use` blocks, and `cache_control` survive the round trip.
    """

    PROVIDER_NAME = "meta"
    ENV_API_KEY_NAME = "MODEL_API_KEY"
    ENV_API_BASE_NAME = "META_API_BASE"
    API_BASE = "https://api.meta.ai/v1"
    PROVIDER_DOCUMENTATION_URL = "https://dev.meta.ai/docs"
    PROMPT_CACHE_KEY_SUPPORT = "supported"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = True
    # reasoning_content is documented as redacted-empty for external callers on Chat
    # Completions; flip to True once verified against the live endpoint.
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = False
    SUPPORTS_MODERATION = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False
    SUPPORTS_IMAGE_GENERATION = False
    SUPPORTS_RERANK = False

    client: AsyncOpenAI
    _anthropic_client: AsyncAnthropic

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        resolved_base = api_base or self.API_BASE
        self.client = AsyncOpenAI(base_url=resolved_base, api_key=api_key, **kwargs)
        # Meta's Anthropic-compatible endpoint expects `Authorization: Bearer`, which the
        # Anthropic SDK only sends via `auth_token`; `api_key` would send `x-api-key` instead.
        self._anthropic_client = AsyncAnthropic(
            auth_token=api_key,
            base_url=_derive_anthropic_base(resolved_base),
            **kwargs,
        )

    @override
    async def _amessages(
        self, params: MessagesParams, **kwargs: Any
    ) -> MessageResponse | ParsedMessage[Any] | ParsedBetaMessage[Any] | AsyncIterator[MessageStreamEvent]:
        """Native Anthropic Messages API pass-through.

        Meta's translation layer does not document support for Anthropic's context
        management or beta primitives, so those are rejected rather than silently dropped.
        """
        if params.context_management is not None or params.betas:
            msg = "context_management and betas are not supported by the Meta Messages API"
            raise NotImplementedError(msg)

        if params.output_format is not None:
            native_kwargs = params.model_dump(
                exclude_none=True, exclude={"output_format", "stream", "betas", "context_management"}
            )
            native_kwargs.update(kwargs)
            if is_structured_output_type(params.output_format):
                return await self._anthropic_client.messages.parse(output_format=params.output_format, **native_kwargs)
            message = await self._anthropic_client.messages.create(
                output_config=cast("Any", params.output_format), **native_kwargs
            )
            return self._convert_native_message_to_response(message)

        api_kwargs = params.model_dump(exclude_none=True, exclude={"betas", "context_management"})
        api_kwargs.pop("stream", None)
        api_kwargs.update(kwargs)

        if params.stream:
            return self._stream_messages_async(**api_kwargs)

        message = await self._anthropic_client.messages.create(**api_kwargs)
        return self._convert_native_message_to_response(message)

    async def _stream_messages_async(self, **kwargs: Any) -> AsyncIterator[MessageStreamEvent]:
        """Stream Meta's native /v1/messages endpoint, yielding typed any-llm event models."""
        async with self._anthropic_client.messages.stream(**kwargs) as stream:
            async for event in stream:
                event_model = _MESSAGE_STREAM_EVENT_TYPES.get(event.type)
                if event_model is not None:
                    yield event_model.model_validate(event, from_attributes=True)

    @staticmethod
    def _convert_native_message_to_response(message: Any) -> MessageResponse:
        return MessageResponse.model_validate(message, from_attributes=True)
