import logging
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock, Mock, patch

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

from any_llm.any_llm import AnyLLM
from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.deepseek.deepseek import DeepseekProvider, _derive_anthropic_base
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


def _build_provider(api_base: str | None = None) -> DeepseekProvider:
    with (
        patch("any_llm.providers.deepseek.deepseek.AsyncOpenAI") as mock_openai,
        patch("any_llm.providers.deepseek.deepseek.AsyncAnthropic") as mock_anthropic,
    ):
        provider = DeepseekProvider(api_key="test-key", api_base=api_base)
        provider._mock_openai_ctor = mock_openai  # type: ignore[attr-defined]
        provider._mock_anthropic_ctor = mock_anthropic  # type: ignore[attr-defined]
    return provider


def _message(text: str = "Hi!") -> Message:
    return Message(
        id="msg_1",
        type="message",
        role="assistant",
        model="deepseek-flash",
        stop_reason="end_turn",
        content=[TextBlock(type="text", text=text)],
        usage=Usage(input_tokens=5, output_tokens=2),
    )


def _params(**overrides: Any) -> MessagesParams:
    fields: dict[str, Any] = {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
    }
    return MessagesParams(**{**fields, **overrides})


def test_provider_declares_images_responses_and_native_messages() -> None:
    metadata = DeepseekProvider.get_provider_metadata()

    assert metadata.image is True
    assert metadata.pdf is False
    assert metadata.responses is True
    assert metadata.messages_native is True
    assert DeepseekProvider.serves_messages_natively("deepseek-v4-pro") is True


@pytest.mark.parametrize(
    ("openai_base", "expected"),
    [
        ("https://api.deepseek.com", "https://api.deepseek.com/anthropic"),
        ("https://api.deepseek.com/", "https://api.deepseek.com/anthropic"),
        ("https://api.deepseek.com/v1", "https://api.deepseek.com/anthropic"),
        ("https://api.deepseek.com/v1/", "https://api.deepseek.com/anthropic"),
        ("https://proxy.example/deepseek", "https://proxy.example/deepseek/anthropic"),
    ],
)
def test_derive_anthropic_base(openai_base: str, expected: str) -> None:
    assert _derive_anthropic_base(openai_base) == expected


def test_init_client_points_both_sdks_at_deepseek() -> None:
    provider = _build_provider()

    provider._mock_openai_ctor.assert_called_once_with(base_url="https://api.deepseek.com", api_key="test-key")  # type: ignore[attr-defined]
    provider._mock_anthropic_ctor.assert_called_once_with(  # type: ignore[attr-defined]
        api_key="test-key", base_url="https://api.deepseek.com/anthropic"
    )


def test_init_client_keeps_openai_only_args_off_the_anthropic_client() -> None:
    with (
        patch("any_llm.providers.deepseek.deepseek.AsyncOpenAI") as mock_openai,
        patch("any_llm.providers.deepseek.deepseek.AsyncAnthropic") as mock_anthropic,
    ):
        DeepseekProvider(api_key="test-key", organization="org-1", project="proj-1", max_retries=3)

    mock_openai.assert_called_once_with(
        base_url="https://api.deepseek.com", api_key="test-key", organization="org-1", project="proj-1", max_retries=3
    )
    mock_anthropic.assert_called_once_with(
        api_key="test-key", base_url="https://api.deepseek.com/anthropic", max_retries=3
    )


def test_init_client_derives_anthropic_base_from_custom_base() -> None:
    provider = _build_provider(api_base="https://proxy.example/v1")

    provider._mock_openai_ctor.assert_called_once_with(base_url="https://proxy.example/v1", api_key="test-key")  # type: ignore[attr-defined]
    provider._mock_anthropic_ctor.assert_called_once_with(  # type: ignore[attr-defined]
        api_key="test-key", base_url="https://proxy.example/anthropic"
    )


@pytest.mark.asyncio
async def test_amessages_non_streaming_uses_native_anthropic_client() -> None:
    provider = _build_provider()
    provider._anthropic_client.messages.create = AsyncMock(return_value=_message())  # type: ignore[method-assign]

    result = await provider._amessages(_params(top_k=5, metadata={"user_id": "u1"}))

    assert isinstance(result, MessageResponse)
    assert result.content[0].text == "Hi!"  # type: ignore[union-attr]
    call_kwargs = provider._anthropic_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "deepseek-flash"
    assert call_kwargs["top_k"] == 5
    assert call_kwargs["metadata"] == {"user_id": "u1"}
    assert "stream" not in call_kwargs
    assert "extra_headers" not in call_kwargs


@pytest.mark.asyncio
async def test_amessages_sends_betas_as_header_and_drops_context_management() -> None:
    provider = _build_provider()
    provider._anthropic_client.messages.create = AsyncMock(return_value=_message())  # type: ignore[method-assign]

    await provider._amessages(
        _params(
            betas=["files-api-2025-04-14", "interleaved-thinking-2025-05-14"],
            context_management={"edits": [{"type": "clear_tool_uses_20250919"}]},
        ),
        extra_headers={"x-trace": "1"},
    )

    call_kwargs = provider._anthropic_client.messages.create.call_args.kwargs
    assert call_kwargs["extra_headers"] == {
        "x-trace": "1",
        "anthropic-beta": "files-api-2025-04-14,interleaved-thinking-2025-05-14",
    }
    assert "betas" not in call_kwargs
    assert "context_management" not in call_kwargs


@pytest.mark.asyncio
async def test_amessages_logs_dropped_context_management(caplog: pytest.LogCaptureFixture) -> None:
    provider = _build_provider()
    provider._anthropic_client.messages.create = AsyncMock(return_value=_message())  # type: ignore[method-assign]

    with caplog.at_level(logging.DEBUG, logger="any_llm"):
        await provider._amessages(_params(context_management={"edits": []}))

    assert "context_management" in caplog.text


@pytest.mark.asyncio
async def test_amessages_refuses_container() -> None:
    provider = _build_provider()
    provider._anthropic_client.messages.create = AsyncMock()  # type: ignore[method-assign]

    with pytest.raises(UnsupportedParameterError, match="container"):
        await provider._amessages(_params(container="container_1"))

    provider._anthropic_client.messages.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_amessages_output_format_goes_through_bridge_without_anthropic_only_fields() -> None:
    class _Answer(BaseModel):
        city: str

    provider = _build_provider()
    provider._anthropic_client.messages.create = AsyncMock()  # type: ignore[method-assign]
    bridged = AsyncMock(return_value=_message('{"city": "Paris"}'))

    with patch.object(AnyLLM, "_amessages", bridged):
        result = await provider._amessages(
            _params(output_format=_Answer, betas=["x"], context_management={"edits": []}),
        )

    assert isinstance(result, Message)
    provider._anthropic_client.messages.create.assert_not_awaited()
    sent: MessagesParams = bridged.call_args.args[0]
    assert sent.output_format is _Answer
    assert sent.betas is None
    assert sent.context_management is None


@pytest.mark.asyncio
async def test_amessages_effort_only_output_format_stays_native() -> None:
    provider = _build_provider()
    provider._anthropic_client.messages.create = AsyncMock(return_value=_message())  # type: ignore[method-assign]
    bridged = AsyncMock()

    with patch.object(AnyLLM, "_amessages", bridged):
        result = await provider._amessages(_params(output_format={"effort": "high"}))

    assert isinstance(result, MessageResponse)
    bridged.assert_not_awaited()
    call_kwargs = provider._anthropic_client.messages.create.call_args.kwargs
    assert call_kwargs["output_config"] == {"effort": "high"}
    assert "output_format" not in call_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output_format",
    [
        {"format": {"type": "json_schema", "schema": {"type": "object"}}, "effort": "high"},
        {"type": "json_schema", "schema": {"type": "object"}},
    ],
)
async def test_amessages_schema_output_format_dict_goes_through_bridge(output_format: dict[str, Any]) -> None:
    provider = _build_provider()
    provider._anthropic_client.messages.create = AsyncMock()  # type: ignore[method-assign]
    bridged = AsyncMock(return_value=_message("{}"))

    with patch.object(AnyLLM, "_amessages", bridged):
        await provider._amessages(_params(output_format=output_format))

    provider._anthropic_client.messages.create.assert_not_awaited()
    assert bridged.call_args.args[0].output_format == output_format


@pytest.mark.asyncio
async def test_amessages_prepares_bridge_blocks_for_native_api() -> None:
    provider = _build_provider()
    provider._anthropic_client.messages.create = AsyncMock(return_value=_message())  # type: ignore[method-assign]

    await provider._amessages(
        _params(
            messages=[
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello", "extra_content": {"deepseek": {}}}],
                },
                {"role": "user", "content": "Again"},
            ]
        )
    )

    sent = provider._anthropic_client.messages.create.call_args.kwargs["messages"]
    assert sent[1]["content"] == [{"type": "text", "text": "Hello"}]


@pytest.mark.asyncio
async def test_amessages_streaming_delegates_to_stream_method() -> None:
    provider = _build_provider()
    provider._stream_messages_async = Mock(return_value=AsyncMock())  # type: ignore[method-assign]

    await provider._amessages(_params(stream=True, betas=["b1"]))

    call_kwargs = provider._stream_messages_async.call_args.kwargs
    assert "stream" not in call_kwargs
    assert call_kwargs["extra_headers"] == {"anthropic-beta": "b1"}


@pytest.mark.asyncio
async def test_stream_messages_async_emits_typed_events() -> None:
    events_list: list[Any] = [
        SDKMessageStartEvent(
            type="message_start",
            message=Message(
                id="msg_123",
                type="message",
                role="assistant",
                content=[],
                model="deepseek-flash",
                stop_reason=None,
                usage=Usage(input_tokens=10, output_tokens=0),
            ),
        ),
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
        event async for event in provider._stream_messages_async(model="deepseek-flash", messages=[], max_tokens=64)
    ]

    assert [type(event) for event in collected] == [
        MessageStartEvent,
        ContentBlockStartEvent,
        ContentBlockDeltaEvent,
        ContentBlockStopEvent,
        MessageDeltaEvent,
        MessageStopEvent,
    ]
    assert collected[2].delta.text == "Hello!"  # type: ignore[union-attr]
