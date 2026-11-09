import json
from collections.abc import AsyncIterator, Callable
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk

from any_llm import AnyLLM
from any_llm.providers.deepinfra.deepinfra import DeepinfraProvider
from any_llm.types.completion import CompletionParams
from any_llm.types.messages import ContentBlockDeltaEvent, TextDelta


def test_provider_basics() -> None:
    """Test provider instantiation and basic attributes."""
    provider = DeepinfraProvider(api_key="dummy_key")
    assert provider.PROVIDER_NAME == "deepinfra"
    assert provider.ENV_API_KEY_NAME == "DEEPINFRA_API_KEY"
    assert provider.ENV_API_BASE_NAME == "DEEPINFRA_API_BASE"
    assert provider.API_BASE == "https://api.deepinfra.com/v1/openai"


def test_deepinfra_supports_flags() -> None:
    """Test that DeepinfraProvider has correct feature support flags."""
    provider = DeepinfraProvider(api_key="dummy_key")
    assert provider.SUPPORTS_COMPLETION is True
    assert provider.SUPPORTS_COMPLETION_STREAMING is True
    assert provider.SUPPORTS_COMPLETION_REASONING is True
    assert provider.SUPPORTS_COMPLETION_IMAGE is True
    assert provider.SUPPORTS_COMPLETION_PDF is False
    assert provider.SUPPORTS_EMBEDDING is True
    assert provider.SUPPORTS_MODERATION is True
    assert provider.SUPPORTS_LIST_MODELS is True
    assert provider.SUPPORTS_VIDEO_GENERATION is True
    assert provider.SUPPORTS_IMAGE_GENERATION is True
    assert provider.SUPPORTS_AUDIO_TRANSCRIPTION is True


def test_factory_integration() -> None:
    """Test that the provider factory can create and discover the provider."""
    provider = AnyLLM.create("deepinfra", api_key="sk-1")
    assert isinstance(provider, DeepinfraProvider)
    assert provider.PROVIDER_NAME == "deepinfra"
    assert "deepinfra" in AnyLLM.get_supported_providers()


def test_deepinfra_convert_completion_params_renames_max_tokens() -> None:
    """max_tokens should be remapped to max_completion_tokens by the base provider."""
    params = CompletionParams(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )
    result = DeepinfraProvider._convert_completion_params(params)
    assert result["max_completion_tokens"] == 100
    assert "max_tokens" not in result


def test_provider_metadata() -> None:
    """Test provider metadata is correctly configured."""
    metadata = DeepinfraProvider.get_provider_metadata()
    assert metadata.name == "deepinfra"
    assert metadata.env_key == "DEEPINFRA_API_KEY"
    assert metadata.env_api_base == "DEEPINFRA_API_BASE"
    assert metadata.doc_url == "https://deepinfra.com/docs/openai_api"
    assert metadata.completion is True


@pytest.mark.asyncio
async def test_deepinfra_streams_messages_output_config_as_json_schema() -> None:
    """A streamed Messages output_config schema reaches DeepInfra as a json_schema response_format.

    DeepInfra serves json_schema on its OpenAI-compatible API, DeepSeek models included, so it
    needs no JSON-mode rewrite. https://docs.deepinfra.com/chat/structured-outputs
    """
    schema = {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}

    async def chunks() -> AsyncIterator[OpenAIChatCompletionChunk]:
        for content, finish_reason in (('{"title": ', None), ('"Session check"}', "stop")):
            yield OpenAIChatCompletionChunk.model_validate(
                {
                    "id": "chunk-1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "deepseek-ai/DeepSeek-V4-Flash",
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}],
                }
            )

    provider = DeepinfraProvider(api_key="dummy_key")
    create = AsyncMock(return_value=chunks())
    with patch.object(provider.client.chat.completions, "create", create):
        stream = await provider.amessages(
            model="deepseek-ai/DeepSeek-V4-Flash",
            messages=[{"role": "user", "content": "Are you there?"}],
            max_tokens=32000,
            output_format={"effort": "high", "format": {"type": "json_schema", "schema": schema}},
            stream=True,
        )
        assert isinstance(stream, AsyncIterator)
        events = [event async for event in stream]

    fragments = [
        event.delta.text
        for event in events
        if isinstance(event, ContentBlockDeltaEvent) and isinstance(event.delta, TextDelta)
    ]
    assert fragments == ['{"title": ', '"Session check"}']
    assert json.loads("".join(fragments)) == {"title": "Session check"}
    assert events[-1].type == "message_stop"
    request = create.call_args.kwargs
    assert request["stream"] is True
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "structured_output", "schema": schema},
    }


def _recording_provider(
    requests: list[httpx.Request], respond: Callable[[httpx.Request], httpx.Response]
) -> DeepinfraProvider:
    """A provider whose HTTP calls are recorded in ``requests`` and answered by ``respond``."""

    def handle(request: httpx.Request) -> httpx.Response:
        request.read()
        requests.append(request)
        return respond(request)

    return DeepinfraProvider(api_key="key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)))


@pytest.mark.asyncio
async def test_image_generation_posts_to_the_openai_images_endpoint() -> None:
    """https://docs.deepinfra.com/api-reference/image-generation/openai-images-generations"""
    requests: list[httpx.Request] = []
    provider = _recording_provider(
        requests, lambda _: httpx.Response(200, json={"created": 1, "data": [{"b64_json": "aW1hZ2U="}]})
    )
    try:
        result = await provider.aimage_generation(
            "black-forest-labs/FLUX-1-schnell",
            "A lighthouse at dusk",
            n=2,
            size="512x512",
            response_format="url",
            quality="hd",
            style="vivid",
            user="user-1",
        )
    finally:
        await provider.client.close()

    assert result.data is not None
    assert result.data[0].b64_json == "aW1hZ2U="
    assert [request.url.path for request in requests] == ["/v1/openai/images/generations"]
    assert json.loads(requests[0].content) == {
        "model": "black-forest-labs/FLUX-1-schnell",
        "prompt": "A lighthouse at dusk",
        "n": 2,
        "size": "512x512",
        "response_format": "url",
        "quality": "hd",
        "style": "vivid",
        "user": "user-1",
    }


@pytest.mark.asyncio
async def test_transcription_uploads_the_clip_with_every_field() -> None:
    """https://docs.deepinfra.com/api-reference/audio/openai-audio-transcriptions"""
    requests: list[httpx.Request] = []
    provider = _recording_provider(requests, lambda _: httpx.Response(200, json={"text": "hello"}))
    try:
        result = await provider.atranscription(
            "openai/whisper-large-v3",
            b"RIFF\x24\x00\x00\x00WAVEfmt ",
            language="en",
            prompt="A greeting.",
            response_format="verbose_json",
            temperature=0.2,
            timestamp_granularities=["word"],
        )
    finally:
        await provider.client.close()

    assert result.text == "hello"
    assert [request.url.path for request in requests] == ["/v1/openai/audio/transcriptions"]
    body = requests[0].content
    assert b'name="file"; filename="audio.wav"\r\nContent-Type: audio/wav\r\n' in body
    for name, value in (
        (b"model", b"openai/whisper-large-v3"),
        (b"language", b"en"),
        (b"prompt", b"A greeting."),
        (b"response_format", b"verbose_json"),
        (b"temperature", b"0.2"),
        (b"timestamp_granularities[]", b"word"),
    ):
        assert b'name="' + name + b'"\r\n\r\n' + value + b"\r\n" in body


@pytest.mark.asyncio
async def test_transcription_wraps_a_plain_text_reply() -> None:
    requests: list[httpx.Request] = []
    provider = _recording_provider(
        requests, lambda _: httpx.Response(200, text="hello", headers={"content-type": "text/plain"})
    )
    try:
        result = await provider.atranscription(
            "openai/whisper-large-v3", b"RIFF\x24\x00\x00\x00WAVEfmt ", response_format="text"
        )
    finally:
        await provider.client.close()

    assert result.text == "hello"
