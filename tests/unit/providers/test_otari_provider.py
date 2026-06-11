from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.exceptions import BatchNotCompleteError
from any_llm.types.audio import AudioSpeechParams, AudioTranscriptionParams
from any_llm.types.batch import BatchResult
from any_llm.types.completion import ChatCompletion, CompletionParams
from any_llm.types.image import ImageGenerationParams
from any_llm.types.model import Model

pytest.importorskip("otari")

from any_llm.providers.otari.otari import (
    OtariProvider,
    _as_plain_dict,
    _extract_model_from_requests,
    _message_stream_event_from_dict,
    _parse_jsonl_to_requests,
)
from any_llm.types.messages import (
    ContentBlockDeltaEvent,
    ContentBlockStartEvent,
    MessageResponse,
    MessagesParams,
    MessageStartEvent,
    MessageStopEvent,
    TextBlock,
)


def _build_provider(mocked_client: MagicMock) -> OtariProvider:
    with patch("any_llm.providers.otari.otari.AsyncOtariClient", return_value=mocked_client):
        return OtariProvider(api_base="https://otari.example.com")


def _mock_otari_client() -> MagicMock:
    client = MagicMock()
    client.platform_mode = False
    client.completion = AsyncMock()
    client.message = AsyncMock()
    client.embedding = AsyncMock()
    client.moderation = AsyncMock()
    client.rerank = AsyncMock()
    client.list_models = AsyncMock()
    client.retrieve_batch = AsyncMock()
    client.cancel_batch = AsyncMock()
    client.list_batches = AsyncMock()
    client.retrieve_batch_results = AsyncMock()
    return client


def test_as_plain_dict_normalizes_models_and_passes_through() -> None:
    from pydantic import BaseModel

    class _Model(BaseModel):
        value: int

    assert _as_plain_dict(_Model(value=1)) == {"value": 1}
    passthrough = {"already": "dict"}
    assert _as_plain_dict(passthrough) is passthrough


def test_extract_model_from_requests_none_when_missing() -> None:
    assert _extract_model_from_requests([]) is None
    assert _extract_model_from_requests([{"custom_id": "1", "body": {}}]) is None


def test_parse_jsonl_to_requests_skips_blank_lines() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n")
        f.write(json.dumps({"custom_id": "r1", "body": {"model": "gpt-4"}}) + "\n")
        path = f.name

    try:
        requests = _parse_jsonl_to_requests(path)
    finally:
        import os

        os.unlink(path)

    assert requests == [{"custom_id": "r1", "body": {"model": "gpt-4"}}]


@patch.dict(
    "os.environ",
    {
        "OTARI_API_BASE": "https://otari.example.com",
        "OTARI_API_KEY": "resolved-api-key",
        "OTARI_PLATFORM_TOKEN": "platform-token",
    },
    clear=False,
)
def test_otari_auto_mode_prefers_explicit_api_key_over_platform_token() -> None:
    with patch("any_llm.providers.otari.otari.AsyncOtariClient") as mock_otari_client:
        mock_otari_client.return_value = _mock_otari_client()
        OtariProvider(api_key="explicit-key")

    call_kwargs = mock_otari_client.call_args.kwargs
    assert call_kwargs["api_key"] == "explicit-key"
    assert "platform_token" not in call_kwargs


@patch.dict(
    "os.environ",
    {
        "OTARI_API_BASE": "https://otari.example.com",
        "OTARI_API_KEY": "resolved-api-key",
        "OTARI_PLATFORM_TOKEN": "",
    },
    clear=False,
)
def test_otari_auto_mode_uses_resolved_api_key_when_no_platform_token() -> None:
    with patch("any_llm.providers.otari.otari.AsyncOtariClient") as mock_otari_client:
        mock_otari_client.return_value = _mock_otari_client()
        OtariProvider()

    call_kwargs = mock_otari_client.call_args.kwargs
    assert call_kwargs["api_key"] == "resolved-api-key"


@pytest.mark.asyncio
async def test_otari_embedding_uses_converter() -> None:
    mocked_client = _mock_otari_client()
    raw_result = {"data": [{"embedding": [0.1], "index": 0}], "model": "m", "object": "list", "usage": {}}
    mocked_client.embedding.return_value = raw_result
    provider = _build_provider(mocked_client)

    sentinel = object()
    with patch.object(provider, "_convert_embedding_response", return_value=sentinel) as mock_convert:
        result = await provider._aembedding("embed-model", "hello", user="u")

    assert result is sentinel
    mocked_client.embedding.assert_awaited_once_with(model="embed-model", input="hello", user="u")
    mock_convert.assert_called_once_with(raw_result)


def test_otari_remaps_max_completion_tokens_to_max_tokens() -> None:
    """The otari gateway accepts ``max_tokens``; the base OpenAI layer remaps it to
    ``max_completion_tokens``, which the gateway errors on, so otari must remap it back."""
    params = CompletionParams(
        model_id="anthropic:claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
    )

    converted = OtariProvider._convert_completion_params(params)

    assert converted["max_tokens"] == 64
    assert "max_completion_tokens" not in converted


def test_otari_converts_pydantic_response_format_to_json_schema() -> None:
    """otari has no parse() helper, so a Pydantic response_format must be sent as a json_schema dict."""
    from pydantic import BaseModel

    class Schema(BaseModel):
        city: str

    params = CompletionParams(
        model_id="anthropic:claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        response_format=Schema,
    )

    converted = OtariProvider._convert_completion_params(params)

    response_format = converted["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "Schema"
    assert "city" in response_format["json_schema"]["schema"]["properties"]


def test_otari_to_parsed_completion_parses_json_content() -> None:
    from pydantic import BaseModel

    from any_llm.types.completion import ParsedChatCompletion

    class Schema(BaseModel):
        city: str

    completion = ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "anthropic:claude-haiku-4-5",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"city": "Paris"}'},
                }
            ],
        }
    )

    parsed = OtariProvider._to_parsed_completion(completion, Schema)

    assert isinstance(parsed, ParsedChatCompletion)
    assert parsed.choices[0].message.parsed == Schema(city="Paris")
    assert parsed.choices[0].message.content == '{"city": "Paris"}'


@pytest.mark.asyncio
async def test_otari_acompletion_returns_parsed_completion_for_structured_output() -> None:
    from pydantic import BaseModel

    from any_llm.types.completion import ParsedChatCompletion

    class Schema(BaseModel):
        city: str

    mocked_client = _mock_otari_client()
    mocked_client.completion = AsyncMock(
        return_value={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "anthropic:claude-haiku-4-5",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": '{"city": "Paris"}'}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    provider = _build_provider(mocked_client)
    params = CompletionParams(
        model_id="anthropic:claude-haiku-4-5",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        response_format=Schema,
    )

    result = await provider._acompletion(params)

    assert isinstance(result, ParsedChatCompletion)
    assert result.choices[0].message.parsed == Schema(city="Paris")
    # The otari client receives a json_schema dict, not the Pydantic class.
    sent = mocked_client.completion.call_args.kwargs["response_format"]
    assert sent["type"] == "json_schema"


@pytest.mark.asyncio
async def test_otari_acompletion_rejects_stream_with_response_format() -> None:
    from pydantic import BaseModel

    class Schema(BaseModel):
        city: str

    provider = _build_provider(_mock_otari_client())
    params = CompletionParams(
        model_id="anthropic:claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        response_format=Schema,
        stream=True,
    )

    with pytest.raises(ValueError, match="stream is not supported for response_format"):
        await provider._acompletion(params)


def test_otari_does_not_advertise_image_or_audio_support() -> None:
    # otari 0.1.0's public async client dropped the OpenAI passthrough that backed
    # these capabilities; they should report as unsupported.
    assert OtariProvider.SUPPORTS_IMAGE_GENERATION is False
    assert OtariProvider.SUPPORTS_AUDIO_TRANSCRIPTION is False
    assert OtariProvider.SUPPORTS_AUDIO_SPEECH is False


@pytest.mark.asyncio
async def test_otari_image_generation_raises_not_implemented() -> None:
    provider = _build_provider(_mock_otari_client())
    params = ImageGenerationParams(model_id="gpt-image", prompt="cat")
    with pytest.raises(NotImplementedError):
        await provider._aimage_generation(params, size="1024x1024")


@pytest.mark.asyncio
async def test_otari_transcription_raises_not_implemented() -> None:
    provider = _build_provider(_mock_otari_client())
    params = AudioTranscriptionParams(model_id="whisper", file=b"audio", language="en")
    with pytest.raises(NotImplementedError):
        await provider._atranscription(params)


@pytest.mark.asyncio
async def test_otari_speech_raises_not_implemented() -> None:
    provider = _build_provider(_mock_otari_client())
    params = AudioSpeechParams(model_id="tts", input="hello", voice="alloy", response_format="mp3")
    with pytest.raises(NotImplementedError):
        await provider._aspeech(params)


@pytest.mark.asyncio
async def test_otari_moderation_uses_default_model_and_converter_include_raw() -> None:
    mocked_client = _mock_otari_client()
    raw_response = {"id": "mod-1"}
    mocked_client.moderation.return_value = raw_response
    provider = _build_provider(mocked_client)

    sentinel = object()
    with patch(
        "any_llm.providers.otari.otari._convert_moderation_response_from_openai", return_value=sentinel
    ) as convert:
        result = await provider._amoderation(model="", input="hello", include_raw=True)

    assert result is sentinel
    mocked_client.moderation.assert_awaited_once_with(model="omni-moderation-latest", input="hello")
    convert.assert_called_once_with(raw_response, include_raw=True)


@pytest.mark.asyncio
async def test_otari_rerank_uses_native_method_and_validates() -> None:
    mocked_client = _mock_otari_client()
    raw_result = {
        "id": "rr-1",
        "results": [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.1}],
    }
    mocked_client.rerank.return_value = raw_result
    provider = _build_provider(mocked_client)

    result = await provider._arerank(model="rerank-1", query="q", documents=["a", "b"], top_n=2)

    assert result.id == "rr-1"
    assert [r.index for r in result.results] == [0, 1]
    mocked_client.rerank.assert_awaited_once_with(model="rerank-1", query="q", documents=["a", "b"], top_n=2)


@pytest.mark.asyncio
async def test_otari_list_models_handles_non_model_items() -> None:
    mocked_client = _mock_otari_client()
    mocked_client.list_models.return_value = [{"id": "m1"}]
    provider = _build_provider(mocked_client)

    validated = MagicMock(spec=Model)
    with patch.object(Model, "model_validate", return_value=validated) as model_validate:
        result = await provider._alist_models()

    assert result == [validated]
    model_validate.assert_called_once_with({"id": "m1"})


@pytest.mark.asyncio
async def test_otari_retrieve_and_cancel_batch_success() -> None:
    mocked_client = _mock_otari_client()
    batch_payload = {
        "id": "batch-1",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "in-1",
        "status": "completed",
        "created_at": 1700000000,
        "completion_window": "24h",
        "request_counts": {"total": 1, "completed": 1, "failed": 0},
    }
    mocked_client.retrieve_batch.return_value = batch_payload
    mocked_client.cancel_batch.return_value = batch_payload
    provider = _build_provider(mocked_client)

    retrieved = await provider._aretrieve_batch("batch-1", provider_name="openai")
    cancelled = await provider._acancel_batch("batch-1", provider_name="openai")

    assert retrieved.id == "batch-1"
    assert cancelled.id == "batch-1"
    mocked_client.retrieve_batch.assert_awaited_once_with(batch_id="batch-1", provider="openai")
    mocked_client.cancel_batch.assert_awaited_once_with(batch_id="batch-1", provider="openai")


@pytest.mark.asyncio
async def test_otari_list_batches_passes_options_when_after_and_limit_set() -> None:
    mocked_client = _mock_otari_client()
    mocked_client.list_batches.return_value = [
        {
            "id": "batch-1",
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "input_file_id": "in-1",
            "status": "completed",
            "created_at": 1700000000,
            "completion_window": "24h",
            "request_counts": {"total": 1, "completed": 1, "failed": 0},
        }
    ]
    provider = _build_provider(mocked_client)

    result = await provider._alist_batches(provider_name="openai", after="cursor-1", limit=5)

    assert result[0].id == "batch-1"
    mocked_client.list_batches.assert_awaited_once_with(provider="openai", options={"after": "cursor-1", "limit": 5})


@pytest.mark.asyncio
async def test_otari_retrieve_batch_results_handles_object_payload_and_error_object() -> None:
    mocked_client = _mock_otari_client()
    payload = SimpleNamespace(
        results=[
            SimpleNamespace(
                custom_id="req-1",
                result=None,
                error=SimpleNamespace(code="bad_request", message="bad"),
            )
        ]
    )
    mocked_client.retrieve_batch_results.return_value = payload
    provider = _build_provider(mocked_client)

    result = await provider._aretrieve_batch_results("batch-1", provider_name="openai")

    assert isinstance(result, BatchResult)
    assert result.results[0].error is not None
    assert result.results[0].error.code == "bad_request"
    assert result.results[0].error.message == "bad"


@pytest.mark.asyncio
async def test_otari_retrieve_batch_results_maps_otari_batch_not_complete_error() -> None:
    mocked_client = _mock_otari_client()
    provider = _build_provider(mocked_client)

    class _FakeOtariBatchNotCompleteError(Exception):
        pass

    mocked_client.retrieve_batch_results.side_effect = _FakeOtariBatchNotCompleteError("pending")

    with patch("any_llm.providers.otari.otari._OTARI_BATCH_NOT_COMPLETE_ERROR", _FakeOtariBatchNotCompleteError):
        with pytest.raises(BatchNotCompleteError):
            await provider._aretrieve_batch_results("batch-1", provider_name="openai")


@pytest.mark.asyncio
async def test_otari_retrieve_batch_results_parses_error_from_dict_item() -> None:
    mocked_client = _mock_otari_client()
    mocked_client.retrieve_batch_results.return_value = {
        "results": [
            {
                "custom_id": "req-1",
                "result": None,
                "error": {"code": "invalid_request", "message": "bad input"},
            }
        ]
    }
    provider = _build_provider(mocked_client)

    result = await provider._aretrieve_batch_results("batch-1", provider_name="openai")

    assert result.results[0].error is not None
    assert result.results[0].error.code == "invalid_request"
    assert result.results[0].error.message == "bad input"


@pytest.mark.asyncio
async def test_otari_completion_sends_max_tokens_not_max_completion_tokens() -> None:
    mocked_client = _mock_otari_client()
    mocked_client.completion = AsyncMock(
        return_value={
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gpt-4",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    with patch("any_llm.providers.otari.otari.AsyncOtariClient", return_value=mocked_client):
        provider = OtariProvider(api_base="https://otari.example.com")

    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=42,
    )

    await provider._acompletion(params)

    call_kwargs = mocked_client.completion.call_args.kwargs
    assert call_kwargs["max_tokens"] == 42
    assert "max_completion_tokens" not in call_kwargs


@pytest.mark.asyncio
async def test_otari_completion_streams_and_converts_chunks() -> None:
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "gpt-4",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
    }

    async def _stream() -> Any:
        yield chunk

    mocked_client = _mock_otari_client()
    mocked_client.completion = AsyncMock(return_value=_stream())
    provider = _build_provider(mocked_client)

    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    )

    result = await provider._acompletion(params)
    collected = [c async for c in result]  # type: ignore[union-attr]

    assert len(collected) == 1
    assert collected[0].choices[0].delta.content == "hi"


def _make_openai_response(text: str):  # type: ignore[no-untyped-def]
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    from any_llm.types.responses import Response

    message = ResponseOutputMessage(
        id="msg-1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )
    return Response(
        id="resp-1",
        created_at=0,
        model="test-model",
        object="response",
        output=[message],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


@pytest.mark.asyncio
async def test_otari_aresponses_basemodel_requests_schema_and_is_parsed() -> None:
    from pydantic import BaseModel

    from any_llm.types.responses import ParsedResponse

    class City(BaseModel):
        city_name: str

    client = _mock_otari_client()
    client.response = AsyncMock(return_value=_make_openai_response('{"city_name": "Paris"}'))
    provider = _build_provider(client)

    result = await provider.aresponses("model", "capital of France?", response_format=City)

    assert "response_format" not in client.response.call_args.kwargs
    assert client.response.call_args.kwargs["text"]["format"]["type"] == "json_schema"
    assert isinstance(result, ParsedResponse)
    assert result.output_parsed is not None
    assert result.output_parsed.city_name == "Paris"


@pytest.mark.asyncio
async def test_otari_aresponses_dict_response_format_sets_text_format() -> None:
    from any_llm.types.responses import ParsedResponse

    client = _mock_otari_client()
    client.response = AsyncMock(return_value=_make_openai_response("{}"))
    provider = _build_provider(client)

    response_format = {"type": "json_schema", "name": "City", "schema": {"type": "object"}}
    result = await provider.aresponses("model", "hi", response_format=response_format)

    assert client.response.call_args.kwargs["text"] == {"format": response_format}
    # A raw dict response_format is passed through unparsed (not a ParsedResponse).
    assert not isinstance(result, ParsedResponse)


@pytest.mark.asyncio
async def test_otari_aresponses_without_response_format() -> None:
    client = _mock_otari_client()
    client.response = AsyncMock(return_value=_make_openai_response("{}"))
    provider = _build_provider(client)

    await provider.aresponses("model", "hi")

    # No structured response_format -> no text.format injected.
    assert "text" not in client.response.call_args.kwargs


def _message_response_payload() -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        # otari-native models carry extra keys; MessageResponse should ignore them.
        "additional_properties": {"served_by": "otari"},
    }


@pytest.mark.asyncio
async def test_otari_amessages_delegates_to_native_endpoint_preserving_anthropic_fields() -> None:
    client = _mock_otari_client()
    client.message.return_value = _message_response_payload()
    provider = _build_provider(client)

    params = MessagesParams(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        system=[{"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}],
        thinking={"type": "enabled", "budget_tokens": 1024},
    )

    result = await provider._amessages(params, metadata={"user_id": "u1"})

    assert isinstance(result, MessageResponse)
    assert result.id == "msg_1"
    block = result.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "hi"

    call_kwargs = client.message.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-5"
    assert call_kwargs["max_tokens"] == 100
    # cache_control on the system block and thinking config survive the pass-through.
    assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call_kwargs["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    # Extra kwargs are forwarded; stream is not set for non-streaming calls.
    assert call_kwargs["metadata"] == {"user_id": "u1"}
    assert "stream" not in call_kwargs


@pytest.mark.asyncio
async def test_otari_amessages_excludes_none_valued_optional_params() -> None:
    client = _mock_otari_client()
    client.message.return_value = _message_response_payload()
    provider = _build_provider(client)

    params = MessagesParams(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )

    await provider._amessages(params)

    call_kwargs = client.message.call_args.kwargs
    # Unset optionals (temperature, tools, ...) are dropped, not sent as None.
    assert "temperature" not in call_kwargs
    assert "tools" not in call_kwargs
    assert "system" not in call_kwargs


@pytest.mark.asyncio
async def test_otari_amessages_streaming_yields_typed_events_and_skips_unknown() -> None:
    raw_events = [
        {"type": "message_start", "message": _message_response_payload()},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "ping"},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
        {"type": "message_stop"},
    ]

    async def _aiter() -> Any:
        for event in raw_events:
            yield event

    client = _mock_otari_client()
    client.message.return_value = _aiter()
    provider = _build_provider(client)

    params = MessagesParams(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
        stream=True,
    )

    result = await provider._amessages(params)

    assert not isinstance(result, MessageResponse)
    collected = [event async for event in result]

    # The unknown "ping" event is skipped; everything else is yielded in order.
    assert isinstance(collected[0], MessageStartEvent)
    assert isinstance(collected[1], ContentBlockStartEvent)
    assert isinstance(collected[2], ContentBlockDeltaEvent)
    assert isinstance(collected[-1], MessageStopEvent)
    assert len(collected) == len(raw_events) - 1
    assert client.message.call_args.kwargs["stream"] is True


def test_message_stream_event_from_dict_returns_none_for_unknown_type() -> None:
    assert _message_stream_event_from_dict({"type": "ping"}) is None
    assert _message_stream_event_from_dict({"no_type": True}) is None
    assert isinstance(_message_stream_event_from_dict({"type": "message_stop"}), MessageStopEvent)
