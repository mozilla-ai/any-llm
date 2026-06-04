from __future__ import annotations

import json
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.exceptions import BatchNotCompleteError
from any_llm.types.audio import AudioSpeechParams, AudioTranscriptionParams
from any_llm.types.batch import BatchResult
from any_llm.types.completion import CompletionParams
from any_llm.types.image import ImageGenerationParams
from any_llm.types.model import Model

pytest.importorskip("otari")

from any_llm.providers.otari.otari import OtariProvider, _extract_model_from_requests, _parse_jsonl_to_requests


def _build_provider(mocked_client: MagicMock) -> OtariProvider:
    with patch("any_llm.providers.otari.otari.OtariClient", return_value=mocked_client):
        return OtariProvider(api_base="https://otari.example.com")


def _mock_otari_client() -> MagicMock:
    client = MagicMock()
    client.platform_mode = False
    client.openai = SimpleNamespace(
        images=SimpleNamespace(generate=AsyncMock()),
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(create=AsyncMock()),
            speech=SimpleNamespace(create=AsyncMock()),
        ),
        moderations=SimpleNamespace(create=AsyncMock()),
    )
    client.embedding = AsyncMock()
    client.list_models = AsyncMock()
    client.retrieve_batch = AsyncMock()
    client.cancel_batch = AsyncMock()
    client.list_batches = AsyncMock()
    client.retrieve_batch_results = AsyncMock()
    return client


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
    with patch("any_llm.providers.otari.otari.OtariClient") as mock_otari_client:
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
    with patch("any_llm.providers.otari.otari.OtariClient") as mock_otari_client:
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


@pytest.mark.asyncio
async def test_otari_image_generation_calls_openai_images_generate() -> None:
    mocked_client = _mock_otari_client()
    mocked_client.openai.images.generate.return_value = MagicMock()
    provider = _build_provider(mocked_client)

    params = ImageGenerationParams(model_id="gpt-image", prompt="cat")
    result = await provider._aimage_generation(params, size="1024x1024")

    assert result is mocked_client.openai.images.generate.return_value
    mocked_client.openai.images.generate.assert_awaited_once_with(model="gpt-image", prompt="cat", size="1024x1024")


@pytest.mark.asyncio
async def test_otari_transcription_calls_openai_audio_transcriptions_create() -> None:
    mocked_client = _mock_otari_client()
    mocked_client.openai.audio.transcriptions.create.return_value = MagicMock()
    provider = _build_provider(mocked_client)

    params = AudioTranscriptionParams(model_id="whisper", file=b"audio", language="en")
    result = await provider._atranscription(params)

    assert result is mocked_client.openai.audio.transcriptions.create.return_value
    mocked_client.openai.audio.transcriptions.create.assert_awaited_once_with(
        model="whisper", file=b"audio", language="en"
    )


@pytest.mark.asyncio
async def test_otari_speech_returns_response_content() -> None:
    mocked_client = _mock_otari_client()
    mocked_client.openai.audio.speech.create.return_value = SimpleNamespace(content=b"audio-bytes")
    provider = _build_provider(mocked_client)

    params = AudioSpeechParams(model_id="tts", input="hello", voice="alloy", response_format="mp3")
    result = await provider._aspeech(params)

    assert result == b"audio-bytes"
    mocked_client.openai.audio.speech.create.assert_awaited_once_with(
        model="tts", input="hello", voice="alloy", response_format="mp3"
    )


@pytest.mark.asyncio
async def test_otari_moderation_uses_default_model_and_converter_include_raw() -> None:
    mocked_client = _mock_otari_client()
    raw_response = {"id": "mod-1"}
    mocked_client.openai.moderations.create.return_value = raw_response
    provider = _build_provider(mocked_client)

    sentinel = object()
    with patch(
        "any_llm.providers.otari.otari._convert_moderation_response_from_openai", return_value=sentinel
    ) as convert:
        result = await provider._amoderation(model="", input="hello", include_raw=True)

    assert result is sentinel
    mocked_client.openai.moderations.create.assert_awaited_once_with(model="omni-moderation-latest", input="hello")
    convert.assert_called_once_with(raw_response, include_raw=True)


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
async def test_otari_completion_uses_converted_params_with_max_tokens_remap() -> None:
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

    with patch("any_llm.providers.otari.otari.OtariClient", return_value=mocked_client):
        provider = OtariProvider(api_base="https://otari.example.com")

    params = CompletionParams(
        model_id="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=42,
    )

    await provider._acompletion(params)

    call_kwargs = mocked_client.completion.call_args.kwargs
    assert call_kwargs["max_completion_tokens"] == 42
    assert "max_tokens" not in call_kwargs


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
