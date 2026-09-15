from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from anthropic.types import (
    ContentBlockDeltaEvent as SDKContentBlockDeltaEvent,
)
from anthropic.types import (
    ContentBlockStartEvent as SDKContentBlockStartEvent,
)
from anthropic.types import (
    ContentBlockStopEvent as SDKContentBlockStopEvent,
)
from anthropic.types import (
    Message,
    MessageDeltaUsage,
    RawMessageDeltaEvent,
    TextBlock,
    TextDelta,
    Usage,
)
from anthropic.types import (
    MessageStartEvent as SDKMessageStartEvent,
)
from anthropic.types import (
    MessageStopEvent as SDKMessageStopEvent,
)
from anthropic.types.raw_message_delta_event import Delta as SDKDelta
from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.meta.meta import MetaProvider, _derive_anthropic_base
from any_llm.types.completion import CompletionParams
from any_llm.types.messages import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    ContentBlockStopEvent,
    MessageDeltaEvent,
    MessageResponse,
    MessagesParams,
    MessageStartEvent,
    MessageStopEvent,
)


def _build_provider(api_base: str | None = None) -> MetaProvider:
    with (
        patch("any_llm.providers.meta.meta.AsyncOpenAI") as mock_openai,
        patch("any_llm.providers.meta.meta.AsyncAnthropic") as mock_anthropic,
    ):
        provider = MetaProvider(api_key="test-key", api_base=api_base)
        provider._mock_openai_ctor = mock_openai  # type: ignore[attr-defined]
        provider._mock_anthropic_ctor = mock_anthropic  # type: ignore[attr-defined]
    return provider


def test_provider_metadata() -> None:
    assert MetaProvider.PROVIDER_NAME == "meta"
    assert MetaProvider.ENV_API_KEY_NAME == "MODEL_API_KEY"
    assert MetaProvider.API_BASE == "https://api.meta.ai/v1"
    assert MetaProvider.PROMPT_CACHE_KEY_SUPPORT == "supported"
    assert MetaProvider.SUPPORTS_COMPLETION is True
    assert MetaProvider.SUPPORTS_COMPLETION_STREAMING is True
    assert MetaProvider.SUPPORTS_RESPONSES is True
    assert MetaProvider.SUPPORTS_MESSAGES is True
    assert MetaProvider.SUPPORTS_COMPLETION_REASONING is False
    assert MetaProvider.SUPPORTS_EMBEDDING is False
    assert MetaProvider.SUPPORTS_BATCH is False
    assert MetaProvider.SUPPORTS_LIST_MODELS is True


def test_derive_anthropic_base_strips_v1_suffix() -> None:
    assert _derive_anthropic_base("https://api.meta.ai/v1") == "https://api.meta.ai"


def test_derive_anthropic_base_leaves_bare_host_untouched() -> None:
    assert _derive_anthropic_base("https://api.meta.ai") == "https://api.meta.ai"


def test_derive_anthropic_base_preserves_custom_override() -> None:
    assert _derive_anthropic_base("https://staging.meta.example/v1") == "https://staging.meta.example"


def test_derive_anthropic_base_strips_trailing_slash_before_v1() -> None:
    """A `.../v1/` override must not leave a trailing slash that Anthropic would double up."""
    assert _derive_anthropic_base("https://custom.meta.example/v1/") == "https://custom.meta.example"


def test_init_client_uses_default_base_urls() -> None:
    provider = _build_provider()

    provider._mock_openai_ctor.assert_called_once_with(base_url="https://api.meta.ai/v1", api_key="test-key")  # type: ignore[attr-defined]
    provider._mock_anthropic_ctor.assert_called_once_with(  # type: ignore[attr-defined]
        auth_token="test-key",  # noqa: S106
        base_url="https://api.meta.ai",
    )


def test_init_client_uses_bearer_auth_token_not_api_key() -> None:
    """Meta expects `Authorization: Bearer`, which the Anthropic SDK only sends via auth_token."""
    provider = _build_provider()

    _, call_kwargs = provider._mock_anthropic_ctor.call_args  # type: ignore[attr-defined]
    assert call_kwargs["auth_token"] == "test-key"  # noqa: S105
    assert "api_key" not in call_kwargs


def test_init_client_derives_anthropic_base_from_custom_openai_base() -> None:
    provider = _build_provider(api_base="https://custom.meta.example/v1")

    provider._mock_openai_ctor.assert_called_once_with(  # type: ignore[attr-defined]
        base_url="https://custom.meta.example/v1", api_key="test-key"
    )
    provider._mock_anthropic_ctor.assert_called_once_with(  # type: ignore[attr-defined]
        auth_token="test-key",  # noqa: S106
        base_url="https://custom.meta.example",
    )


@pytest.mark.asyncio
async def test_acompletion_delegates_to_openai_client() -> None:
    provider = _build_provider()
    mock_response = Mock()
    provider.client.chat.completions.create = AsyncMock(return_value=mock_response)  # type: ignore[method-assign]

    params = CompletionParams(model_id="muse-spark-1.2", messages=[{"role": "user", "content": "hi"}], stream=False)
    with patch.object(MetaProvider, "_convert_completion_response_async", return_value=mock_response):
        result = await provider._acompletion(params)

    assert result is mock_response
    provider.client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_amessages_non_streaming_uses_native_anthropic_client() -> None:
    provider = _build_provider()
    mock_message = Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="muse-spark-1.2",
        stop_reason="end_turn",
        content=[TextBlock(type="text", text="Hi!")],
        usage=Usage(input_tokens=5, output_tokens=2),
    )
    provider._anthropic_client.messages.create = AsyncMock(return_value=mock_message)  # type: ignore[method-assign]

    params = MessagesParams(
        model="muse-spark-1.2",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64,
    )
    result = await provider._amessages(params)

    assert isinstance(result, MessageResponse)
    assert result.content[0].text == "Hi!"  # type: ignore[union-attr]
    provider._anthropic_client.messages.create.assert_awaited_once()
    call_kwargs = provider._anthropic_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "muse-spark-1.2"
    assert "stream" not in call_kwargs


@pytest.mark.asyncio
async def test_amessages_rejects_context_management() -> None:
    provider = _build_provider()
    params = MessagesParams(
        model="muse-spark-1.2",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64,
        context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
    )
    with pytest.raises(NotImplementedError, match="context_management"):
        await provider._amessages(params)


@pytest.mark.asyncio
async def test_amessages_rejects_betas() -> None:
    provider = _build_provider()
    params = MessagesParams(
        model="muse-spark-1.2",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64,
        betas=["some-beta"],
    )
    with pytest.raises(NotImplementedError, match="betas"):
        await provider._amessages(params)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("container", "container_123"),
        ("prompt_cache_key", "my-app"),
        ("stop_sequences", ["STOP"]),
        ("top_k", 5),
    ],
)
@pytest.mark.asyncio
async def test_amessages_rejects_unsupported_params(field_name: str, value: Any) -> None:
    """Reject fields that Meta's Messages endpoint rejects or does not document."""
    provider = _build_provider()
    params = MessagesParams(
        model="muse-spark-1.2",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64,
        **{field_name: value},
    )
    with pytest.raises(UnsupportedParameterError, match=field_name):
        await provider._amessages(params)
    provider._anthropic_client.messages.create.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_amessages_output_format_rejects_stream() -> None:
    """The public `amessages()` already blocks this combo before building params; `_amessages`
    guards it too so a direct call (e.g. bypassing the public entry point) fails loudly instead
    of silently dropping `stream` and returning a non-streaming result."""

    class _Answer(BaseModel):
        text: str

    provider = _build_provider()
    params = MessagesParams(
        model="muse-spark-1.2",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64,
        output_format=_Answer,
        stream=True,
    )
    with pytest.raises(ValueError, match="stream is not supported for output_format"):
        await provider._amessages(params)


@pytest.mark.asyncio
async def test_amessages_streaming_delegates_to_stream_method() -> None:
    provider = _build_provider()
    provider._stream_messages_async = Mock(return_value=AsyncMock())  # type: ignore[method-assign]

    params = MessagesParams(
        model="muse-spark-1.2",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64,
        stream=True,
    )
    await provider._amessages(params)
    provider._stream_messages_async.assert_called_once()


@pytest.mark.asyncio
async def test_stream_messages_async_emits_typed_events() -> None:
    usage_start = Usage(input_tokens=10, output_tokens=0)
    msg = Message(
        id="msg_123",
        type="message",
        role="assistant",
        content=[],
        model="muse-spark-1.2",
        stop_reason=None,
        usage=usage_start,
    )

    events_list: list[Any] = [
        SDKMessageStartEvent(type="message_start", message=msg),
        # Anthropic streams send periodic `ping` events with no any-llm equivalent; the
        # generator must skip them rather than yield the raw SDK event or raise.
        SimpleNamespace(type="ping"),
        SDKContentBlockStartEvent(type="content_block_start", index=0, content_block=TextBlock(type="text", text="")),
        SDKContentBlockDeltaEvent(
            type="content_block_delta", index=0, delta=TextDelta(type="text_delta", text="Hello!")
        ),
        SDKContentBlockStopEvent(type="content_block_stop", index=0),
        RawMessageDeltaEvent(
            type="message_delta",
            delta=SDKDelta(stop_reason="end_turn"),
            usage=MessageDeltaUsage(output_tokens=5),
        ),
        SDKMessageStopEvent(type="message_stop"),
    ]

    class MockStream:
        def __init__(self) -> None:
            self.events = iter(events_list)

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

        def __aiter__(self) -> Self:
            return self

        async def __anext__(self) -> Any:
            try:
                return next(self.events)
            except StopIteration:
                raise StopAsyncIteration from None

    provider = _build_provider()
    provider._anthropic_client.messages.stream = Mock(return_value=MockStream())  # type: ignore[method-assign]

    collected = [
        event async for event in provider._stream_messages_async(model="muse-spark-1.2", messages=[], max_tokens=64)
    ]

    # The `ping` event has no any-llm equivalent and must be dropped, not yielded verbatim.
    assert len(collected) == 6
    assert isinstance(collected[0], MessageStartEvent)
    assert collected[0].message.id == "msg_123"
    assert isinstance(collected[1], ContentBlockStartEvent)
    assert collected[1].content_block.type == "text"
    assert isinstance(collected[2], ContentBlockDeltaEvent)
    assert collected[2].delta.text == "Hello!"  # type: ignore[union-attr]
    assert isinstance(collected[3], ContentBlockStopEvent)
    assert isinstance(collected[4], MessageDeltaEvent)
    assert collected[4].delta.stop_reason == "end_turn"
    assert collected[4].usage.output_tokens == 5
    assert isinstance(collected[5], MessageStopEvent)


@pytest.mark.asyncio
async def test_amessages_output_format_pydantic_type_uses_parse() -> None:
    class _Answer(BaseModel):
        text: str

    provider = _build_provider()
    parsed_result = Mock()
    provider._anthropic_client.messages.parse = AsyncMock(return_value=parsed_result)  # type: ignore[method-assign]

    params = MessagesParams(
        model="muse-spark-1.2",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64,
        output_format=_Answer,
    )
    result = await provider._amessages(params)

    assert result is parsed_result
    call_kwargs = provider._anthropic_client.messages.parse.call_args.kwargs
    assert call_kwargs["output_format"] is _Answer
    assert "stream" not in call_kwargs


@pytest.mark.asyncio
async def test_amessages_output_format_dict_uses_output_config() -> None:
    provider = _build_provider()
    mock_message = Message(
        id="msg_2",
        type="message",
        role="assistant",
        model="muse-spark-1.2",
        stop_reason="end_turn",
        content=[TextBlock(type="text", text="{}")],
        usage=Usage(input_tokens=1, output_tokens=1),
    )
    provider._anthropic_client.messages.create = AsyncMock(return_value=mock_message)  # type: ignore[method-assign]

    output_config = {"format": {"type": "json_schema", "schema": {"type": "object"}}}
    params = MessagesParams(
        model="muse-spark-1.2",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=64,
        output_format=output_config,
    )
    result = await provider._amessages(params)

    assert isinstance(result, MessageResponse)
    call_kwargs = provider._anthropic_client.messages.create.call_args.kwargs
    assert call_kwargs["output_config"] == output_config


@pytest.mark.asyncio
async def test_alist_models_delegates_to_openai_client() -> None:
    provider = _build_provider()
    fake_response = MagicMock()
    fake_response.data = []
    provider.client.models.list = AsyncMock(return_value=fake_response)  # type: ignore[method-assign]

    result = await provider._alist_models()

    assert result == []
    provider.client.models.list.assert_awaited_once()
