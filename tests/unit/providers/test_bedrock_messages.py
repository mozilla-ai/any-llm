"""Tests for the Bedrock provider's native Messages path for Claude models."""

import base64
import io
import json
import struct
import zlib
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from botocore.awsrequest import AWSPreparedRequest, AWSResponse
from botocore.exceptions import EventStreamError
from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.bedrock import BedrockProvider
from any_llm.types.messages import (
    ContentBlockDeltaEvent,
    MessageDeltaEvent,
    MessageResponse,
    MessageStartEvent,
    MessageStopEvent,
    ParsedMessage,
    ThinkingBlock,
)

CLAUDE_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

_CACHED_SYSTEM: list[dict[str, Any]] = [
    {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
]

_THINKING_HISTORY: list[dict[str, Any]] = [
    {"role": "user", "content": "What is 2+2?"},
    {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "Add them.", "signature": "sig-1"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "4"},
        ],
    },
    {"role": "user", "content": [{"type": "text", "text": "And 3+3?", "cache_control": {"type": "ephemeral"}}]},
]

_CLAUDE_MESSAGE: dict[str, Any] = {
    "id": "msg_bdrk_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5-20251001",
    "content": [
        {"type": "thinking", "thinking": "Add again.", "signature": "sig-2"},
        {"type": "text", "text": "6"},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 12,
        "output_tokens": 7,
        "cache_creation_input_tokens": 1024,
        "cache_read_input_tokens": 2048,
    },
}

_CLAUDE_STREAM: list[dict[str, Any]] = [
    {"type": "message_start", "message": {**_CLAUDE_MESSAGE, "content": [], "stop_reason": None}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Add again."}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig-2"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "ping"},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "6"}},
    {"type": "content_block_stop", "index": 1},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 7},
    },
    {
        "type": "message_stop",
        "amazon-bedrock-invocationMetrics": {"inputTokenCount": 12, "outputTokenCount": 7},
    },
]


class _FakeEventStream:
    """Stands in for botocore's ``EventStream``: iterable once, and closable."""

    def __init__(self, events: list[dict[str, Any]], error: Exception | None = None) -> None:
        self._events = events
        self._error = error
        self.closed = False

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for event in self._events:
            yield {"chunk": {"bytes": json.dumps(event).encode()}}
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self.closed = True


async def _drain(stream: AsyncIterator[Any], received: list[Any]) -> None:
    async for event in stream:
        received.append(event)


def _custom_client_provider(response_body: dict[str, Any] | None = None) -> tuple[BedrockProvider, Mock]:
    client = Mock()
    client.invoke_model.return_value = {"body": Mock(read=Mock(return_value=json.dumps(response_body or {}).encode()))}
    return BedrockProvider(client=client), client


def _sent_body(client_method: Mock) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(client_method.call_args.kwargs["body"])
    return body


@pytest.mark.asyncio
async def test_claude_request_sends_the_anthropic_body_with_cache_control_thinking_and_betas() -> None:
    provider, client = _custom_client_provider(_CLAUDE_MESSAGE)

    await provider.amessages(
        model=CLAUDE_MODEL,
        messages=_THINKING_HISTORY,
        system=_CACHED_SYSTEM,
        max_tokens=2048,
        thinking={"type": "enabled", "budget_tokens": 1024},
        betas=["interleaved-thinking-2025-05-14"],
        service_tier="auto",
        guardrailIdentifier="gr-1",
    )

    client.converse.assert_not_called()
    call_kwargs = client.invoke_model.call_args.kwargs
    assert call_kwargs["modelId"] == CLAUDE_MODEL
    assert call_kwargs["contentType"] == "application/json"
    assert call_kwargs["accept"] == "application/json"
    assert call_kwargs["guardrailIdentifier"] == "gr-1"
    assert _sent_body(client.invoke_model) == {
        "anthropic_version": "bedrock-2023-05-31",
        "anthropic_beta": ["interleaved-thinking-2025-05-14"],
        "messages": _THINKING_HISTORY,
        "system": _CACHED_SYSTEM,
        "max_tokens": 2048,
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }


@pytest.mark.asyncio
async def test_claude_response_keeps_thinking_signatures_and_cache_usage() -> None:
    provider, _ = _custom_client_provider(_CLAUDE_MESSAGE)

    result = await provider.amessages(model=CLAUDE_MODEL, messages=_THINKING_HISTORY, max_tokens=2048)

    assert isinstance(result, MessageResponse)
    thinking = result.content[0]
    assert isinstance(thinking, ThinkingBlock)
    assert thinking.signature == "sig-2"
    assert result.usage.cache_creation_input_tokens == 1024
    assert result.usage.cache_read_input_tokens == 2048
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_claude_request_without_betas_sends_no_beta_field() -> None:
    provider, client = _custom_client_provider(_CLAUDE_MESSAGE)

    await provider.amessages(model=CLAUDE_MODEL, messages=_THINKING_HISTORY, max_tokens=64)

    assert "anthropic_beta" not in _sent_body(client.invoke_model)


@pytest.mark.asyncio
async def test_claude_context_management_carries_its_inferred_beta() -> None:
    provider, client = _custom_client_provider(_CLAUDE_MESSAGE)
    context_management = {"edits": [{"type": "clear_tool_uses_20250919"}]}

    await provider.amessages(
        model=CLAUDE_MODEL, messages=_THINKING_HISTORY, max_tokens=64, context_management=context_management
    )

    body = _sent_body(client.invoke_model)
    assert body["context_management"] == context_management
    assert body["anthropic_beta"] == ["context-management-2025-06-27"]


@pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    ],
)
@pytest.mark.asyncio
async def test_claude_model_id_forms_take_the_native_path(model: str) -> None:
    provider, client = _custom_client_provider(_CLAUDE_MESSAGE)

    await provider.amessages(model=model, messages=_THINKING_HISTORY, max_tokens=64)

    assert client.invoke_model.call_args.kwargs["modelId"] == model


@pytest.mark.asyncio
async def test_claude_stream_yields_the_anthropic_events_and_closes_the_stream() -> None:
    event_stream = _FakeEventStream(_CLAUDE_STREAM)
    client = Mock()
    client.invoke_model_with_response_stream.return_value = {"body": event_stream}
    provider = BedrockProvider(client=client)

    stream = await provider.amessages(
        model=CLAUDE_MODEL,
        messages=_THINKING_HISTORY,
        max_tokens=2048,
        stream=True,
        betas=["interleaved-thinking-2025-05-14"],
    )
    assert isinstance(stream, AsyncIterator)
    events = [event async for event in stream]

    assert "stream" not in _sent_body(client.invoke_model_with_response_stream)
    assert _sent_body(client.invoke_model_with_response_stream)["anthropic_beta"] == ["interleaved-thinking-2025-05-14"]
    assert [event.type for event in events] == [event["type"] for event in _CLAUDE_STREAM if event["type"] != "ping"]
    assert isinstance(events[0], MessageStartEvent)
    assert events[0].message.usage.cache_read_input_tokens == 2048
    signature_delta = events[3]
    assert isinstance(signature_delta, ContentBlockDeltaEvent)
    assert signature_delta.delta.type == "signature_delta"
    assert signature_delta.delta.signature == "sig-2"
    assert isinstance(events[-2], MessageDeltaEvent)
    assert events[-2].delta.stop_reason == "end_turn"
    assert isinstance(events[-1], MessageStopEvent)
    assert event_stream.closed


@pytest.mark.asyncio
async def test_claude_stream_error_propagates_and_closes_the_stream() -> None:
    error = EventStreamError(  # type: ignore[no-untyped-call]
        {"Error": {"Code": "throttlingException", "Message": "Too many requests"}}, "InvokeModelWithResponseStream"
    )
    event_stream = _FakeEventStream(_CLAUDE_STREAM[:2], error=error)
    client = Mock()
    client.invoke_model_with_response_stream.return_value = {"body": event_stream}
    provider = BedrockProvider(client=client)

    stream = await provider.amessages(model=CLAUDE_MODEL, messages=_THINKING_HISTORY, max_tokens=64, stream=True)
    assert isinstance(stream, AsyncIterator)
    received: list[Any] = []
    with pytest.raises(EventStreamError, match="Too many requests"):
        await _drain(stream, received)

    assert [event.type for event in received] == ["message_start", "content_block_start"]
    assert event_stream.closed


@pytest.mark.asyncio
async def test_claude_stream_closed_early_closes_the_bedrock_stream() -> None:
    event_stream = _FakeEventStream(_CLAUDE_STREAM)

    stream = cast("AsyncGenerator[Any]", BedrockProvider._stream_native_messages(event_stream))
    first = await anext(stream)
    await stream.aclose()

    assert isinstance(first, MessageStartEvent)
    assert event_stream.closed


@pytest.mark.asyncio
async def test_non_claude_model_keeps_the_converse_bridge() -> None:
    client = Mock()
    client.converse.return_value = {
        "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4},
    }
    provider = BedrockProvider(client=client)

    result = await provider.amessages(
        model="amazon.nova-lite-v1:0", messages=[{"role": "user", "content": "Hi"}], max_tokens=64
    )

    client.invoke_model.assert_not_called()
    assert client.converse.call_args.kwargs["modelId"] == "amazon.nova-lite-v1:0"
    assert isinstance(result, MessageResponse)
    assert result.content[0].type == "text"


@pytest.mark.asyncio
async def test_non_claude_model_still_refuses_betas() -> None:
    provider = BedrockProvider(client=Mock())

    with pytest.raises(NotImplementedError, match="betas"):
        await provider.amessages(
            model="amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=64,
            betas=["interleaved-thinking-2025-05-14"],
        )


@pytest.mark.asyncio
async def test_claude_output_format_keeps_the_converse_bridge() -> None:
    class City(BaseModel):
        city: str

    client = Mock()
    client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "t1", "name": "any_llm_structured_output", "input": {"city": "Paris"}}}
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 3, "outputTokens": 1, "totalTokens": 4},
    }
    provider = BedrockProvider(client=client)

    result = await provider.amessages(
        model=CLAUDE_MODEL,
        messages=[{"role": "user", "content": "Capital of France?"}],
        max_tokens=64,
        output_format=City,
    )

    client.invoke_model.assert_not_called()
    assert isinstance(result, ParsedMessage)
    assert result.parsed_output == City(city="Paris")


@pytest.mark.asyncio
async def test_claude_timeout_uses_a_client_built_with_that_timeout() -> None:
    with patch("boto3.Session") as session_cls:
        session_client = session_cls.return_value.client
        timeout_client = Mock()
        timeout_client.invoke_model.return_value = {"body": Mock(read=Mock(return_value=json.dumps(_CLAUDE_MESSAGE)))}
        session_client.side_effect = [Mock(), timeout_client]
        provider = BedrockProvider(api_key="token")

        await provider.amessages(model=CLAUDE_MODEL, messages=_THINKING_HISTORY, max_tokens=64, timeout=7)

    timeout_config = session_client.call_args.kwargs["config"]
    assert timeout_config.connect_timeout == 7
    assert timeout_config.read_timeout == 7
    assert "timeout" not in timeout_client.invoke_model.call_args.kwargs


@pytest.mark.asyncio
async def test_claude_timeout_with_custom_client_is_refused() -> None:
    provider, client = _custom_client_provider(_CLAUDE_MESSAGE)

    with pytest.raises(UnsupportedParameterError, match="timeout"):
        await provider.amessages(model=CLAUDE_MODEL, messages=_THINKING_HISTORY, max_tokens=64, timeout=7)

    client.invoke_model.assert_not_called()


def test_serves_messages_natively_only_for_claude_models() -> None:
    assert BedrockProvider.SUPPORTS_MESSAGES_NATIVE is True
    assert BedrockProvider.serves_messages_natively(CLAUDE_MODEL) is True
    assert BedrockProvider.serves_messages_natively("anthropic.claude-3-haiku-20240307-v1:0") is True
    assert BedrockProvider.serves_messages_natively("amazon.nova-lite-v1:0") is False
    assert BedrockProvider.serves_messages_natively("meta.llama3-70b-instruct-v1:0") is False


def _event_stream_message(headers: dict[str, str], payload: bytes) -> bytes:
    """Encode one ``application/vnd.amazon.eventstream`` message, as Bedrock sends it."""
    encoded_headers = b"".join(
        struct.pack(">B", len(name)) + name.encode() + struct.pack(">BH", 7, len(value)) + value.encode()
        for name, value in headers.items()
    )
    prelude = struct.pack(">II", 16 + len(encoded_headers) + len(payload), len(encoded_headers))
    message = prelude + struct.pack(">I", zlib.crc32(prelude)) + encoded_headers + payload
    return message + struct.pack(">I", zlib.crc32(message))


def _chunk_message(event: dict[str, Any]) -> bytes:
    payload = json.dumps({"bytes": base64.b64encode(json.dumps(event).encode()).decode()}).encode()
    headers = {":event-type": "chunk", ":content-type": "application/json", ":message-type": "event"}
    return _event_stream_message(headers, payload)


class _RawBody(io.BytesIO):
    """The urllib3 response surface botocore reads: ``read`` for a blob, ``stream`` for events."""

    def stream(self, **kwargs: Any) -> Iterator[bytes]:
        yield self.read()


def _capture_requests(provider: BedrockProvider, response_body: bytes) -> list[AWSPreparedRequest]:
    """Answer every request the provider's real botocore client sends with ``response_body``."""
    sent: list[AWSPreparedRequest] = []

    def respond(request: AWSPreparedRequest, **kwargs: Any) -> AWSResponse:
        sent.append(request)
        return AWSResponse(request.url, 200, {}, _RawBody(response_body))  # type: ignore[no-untyped-call]

    provider.client.meta.events.register("before-send", respond)
    return sent


@pytest.fixture
def isolated_aws_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> pytest.MonkeyPatch:
    """Keep the developer's own AWS credentials and config out of the auth tests."""
    for name in (
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    return monkeypatch


@pytest.mark.usefixtures("isolated_aws_env")
@pytest.mark.asyncio
async def test_claude_request_signs_with_the_bearer_api_key() -> None:
    provider = BedrockProvider(api_key="bedrock-api-key", region_name="us-east-1")
    sent = _capture_requests(provider, json.dumps(_CLAUDE_MESSAGE).encode())

    result = await provider.amessages(model=CLAUDE_MODEL, messages=_THINKING_HISTORY, max_tokens=64)

    assert isinstance(result, MessageResponse)
    (request,) = sent
    assert request.headers["Authorization"] == b"Bearer bedrock-api-key"
    assert request.url == (
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-haiku-4-5-20251001-v1%3A0/invoke"
    )
    assert json.loads(request.body)["messages"] == _THINKING_HISTORY


@pytest.mark.asyncio
async def test_claude_request_signs_with_the_bearer_token_from_the_environment(
    isolated_aws_env: pytest.MonkeyPatch,
) -> None:
    isolated_aws_env.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bearer-token")
    provider = BedrockProvider(region_name="us-east-1")
    sent = _capture_requests(provider, json.dumps(_CLAUDE_MESSAGE).encode())

    await provider.amessages(model=CLAUDE_MODEL, messages=_THINKING_HISTORY, max_tokens=64)

    assert sent[0].headers["Authorization"] == b"Bearer env-bearer-token"


@pytest.mark.usefixtures("isolated_aws_env")
@pytest.mark.asyncio
async def test_claude_request_signs_with_explicit_iam_keys() -> None:
    provider = BedrockProvider(
        aws_access_key_id="AKIAEXPLICIT",
        aws_secret_access_key="explicit-secret",  # noqa: S106
        region_name="us-east-1",
    )
    sent = _capture_requests(provider, json.dumps(_CLAUDE_MESSAGE).encode())

    await provider.amessages(model=CLAUDE_MODEL, messages=_THINKING_HISTORY, max_tokens=64)

    authorization = sent[0].headers["Authorization"].decode()
    assert authorization.startswith("AWS4-HMAC-SHA256 Credential=AKIAEXPLICIT/")
    assert "/us-east-1/bedrock/aws4_request" in authorization


@pytest.mark.asyncio
async def test_claude_request_signs_with_ambient_credentials(isolated_aws_env: pytest.MonkeyPatch) -> None:
    isolated_aws_env.setenv("AWS_ACCESS_KEY_ID", "AKIAAMBIENT")
    isolated_aws_env.setenv("AWS_SECRET_ACCESS_KEY", "ambient-secret")
    provider = BedrockProvider(region_name="us-west-2")
    sent = _capture_requests(provider, json.dumps(_CLAUDE_MESSAGE).encode())

    await provider.amessages(model=CLAUDE_MODEL, messages=_THINKING_HISTORY, max_tokens=64)

    authorization = sent[0].headers["Authorization"].decode()
    assert authorization.startswith("AWS4-HMAC-SHA256 Credential=AKIAAMBIENT/")
    assert sent[0].url.startswith("https://bedrock-runtime.us-west-2.amazonaws.com/")


@pytest.mark.usefixtures("isolated_aws_env")
@pytest.mark.asyncio
async def test_claude_stream_parses_a_real_bedrock_event_stream() -> None:
    provider = BedrockProvider(api_key="bedrock-api-key", region_name="us-east-1")
    sent = _capture_requests(provider, b"".join(_chunk_message(event) for event in _CLAUDE_STREAM))

    stream = await provider.amessages(model=CLAUDE_MODEL, messages=_THINKING_HISTORY, max_tokens=64, stream=True)
    assert isinstance(stream, AsyncIterator)
    events = [event async for event in stream]

    assert sent[0].url.endswith("/invoke-with-response-stream")
    assert sent[0].headers["Authorization"] == b"Bearer bedrock-api-key"
    assert [event.type for event in events] == [event["type"] for event in _CLAUDE_STREAM if event["type"] != "ping"]


@pytest.mark.usefixtures("isolated_aws_env")
@pytest.mark.asyncio
async def test_claude_stream_raises_a_real_bedrock_error_event() -> None:
    provider = BedrockProvider(api_key="bedrock-api-key", region_name="us-east-1")
    error_event = _event_stream_message(
        {
            ":exception-type": "throttlingException",
            ":content-type": "application/json",
            ":message-type": "exception",
        },
        json.dumps({"message": "Too many requests"}).encode(),
    )
    _capture_requests(provider, _chunk_message(_CLAUDE_STREAM[0]) + error_event)

    stream = await provider.amessages(model=CLAUDE_MODEL, messages=_THINKING_HISTORY, max_tokens=64, stream=True)
    assert isinstance(stream, AsyncIterator)
    received: list[Any] = []
    with pytest.raises(EventStreamError, match="Too many requests"):
        await _drain(stream, received)

    assert [event.type for event in received] == ["message_start"]
