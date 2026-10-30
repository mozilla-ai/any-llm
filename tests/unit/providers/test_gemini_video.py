from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from google.genai._gaos.types.interactions import Interaction

from any_llm.exceptions import InvalidRequestError, ProviderError, UnsupportedParameterError
from any_llm.providers.gemini import GeminiProvider
from any_llm.providers.gemini.video import convert_interaction_to_video_job, convert_video_params
from any_llm.types.video import VideoGenerationParams, VideoJob, VideoOutput

MODEL = "gemini-omni-1.1-flash"
VIDEO_B64 = "dmlkZW8="


def _interaction(status: str = "in_progress", **overrides: Any) -> Interaction:
    fields: dict[str, Any] = {
        "id": "int-123",
        "object": "interaction",
        "model": MODEL,
        "status": status,
        "created": "2026-10-30T10:00:00Z",
        "steps": [],
    }
    fields.update(overrides)
    return Interaction.model_validate(fields)


def _video_step(**content: Any) -> dict[str, Any]:
    return {"type": "model_output", "content": [{"type": "video", **content}]}


def _params(**overrides: Any) -> VideoGenerationParams:
    fields: dict[str, Any] = {"model_id": MODEL, "prompt": "a paper boat"}
    fields.update(overrides)
    return VideoGenerationParams(**fields)


def make_provider() -> tuple[GeminiProvider, Mock]:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        provider = GeminiProvider(api_key="test-key")
    return provider, client


def test_convert_video_params_runs_a_text_prompt_in_the_background() -> None:
    assert convert_video_params(_params(), "gemini", api_version=None) == {
        "model": MODEL,
        "input": "a paper boat",
        "background": True,
        "response_format": {"type": "video"},
    }


def test_convert_video_params_forwards_the_clients_api_version_and_seed() -> None:
    converted = convert_video_params(_params(seed=7), "gemini", api_version="v1beta")
    assert converted["api_version"] == "v1beta"
    assert converted["generation_config"] == {"seed": 7}


@pytest.mark.parametrize(
    ("size", "aspect_ratio", "resolution"),
    [
        ("640x360", "16:9", "360p"),
        ("1280x720", "16:9", "720p"),
        ("720x1280", "9:16", "720p"),
        ("1920x1080", "16:9", "1080p"),
        ("2160x3840", "9:16", "4k"),
    ],
)
def test_convert_video_params_maps_size_to_aspect_ratio_and_resolution(
    size: str, aspect_ratio: str, resolution: str
) -> None:
    converted = convert_video_params(_params(size=size), "gemini", api_version=None)
    assert converted["response_format"] == {"type": "video", "aspect_ratio": aspect_ratio, "resolution": resolution}


def test_convert_video_params_rejects_a_square_size() -> None:
    with pytest.raises(InvalidRequestError, match="cannot be square"):
        convert_video_params(_params(size="720x720"), "gemini", api_version=None)


def test_convert_video_params_rejects_a_size_outside_geminis_resolutions() -> None:
    with pytest.raises(InvalidRequestError, match="short side of 360, 720, 1080, 2160 pixels, not '864x480'"):
        convert_video_params(_params(size="864x480"), "gemini", api_version=None)


@pytest.mark.parametrize("field", ["seconds", "negative_prompt"])
def test_convert_video_params_rejects_fields_gemini_has_no_home_for(field: str) -> None:
    value: Any = 5 if field == "seconds" else "blurry"
    with pytest.raises(UnsupportedParameterError, match=field):
        convert_video_params(_params(**{field: value}), "gemini", api_version=None)


def test_convert_video_params_puts_frames_then_references_before_the_prompt() -> None:
    params = _params(
        first_frame="https://img/first.png",
        last_frame="data:image/jpeg;base64,Zmlyc3Q=",
        reference_images=["https://generativelanguage.googleapis.com/v1beta/files/ref"],
    )
    converted = convert_video_params(params, "gemini", api_version=None)
    assert converted["input"] == [
        {"type": "image", "uri": "https://img/first.png"},
        {"type": "image", "data": "Zmlyc3Q=", "mime_type": "image/jpeg"},
        {"type": "image", "uri": "https://generativelanguage.googleapis.com/v1beta/files/ref"},
        {"type": "text", "text": "a paper boat"},
    ]


def test_convert_video_params_keeps_the_prompt_unchanged_with_a_single_frame() -> None:
    converted = convert_video_params(_params(last_frame="https://img/last.png"), "gemini", api_version=None)
    assert converted["input"] == [
        {"type": "image", "uri": "https://img/last.png"},
        {"type": "text", "text": "a paper boat"},
    ]


@pytest.mark.parametrize("image", ["data:image/png,Zmlyc3Q=", "data:;base64,Zmlyc3Q=", "data:image/png;base64"])
def test_convert_video_params_rejects_malformed_data_urls(image: str) -> None:
    with pytest.raises(InvalidRequestError, match="data:image/png;base64"):
        convert_video_params(_params(first_frame=image), "gemini", api_version=None)


def test_convert_interaction_maps_a_running_job() -> None:
    result = convert_interaction_to_video_job(_interaction(), fallback_model="other")

    assert isinstance(result, VideoJob)
    assert result.id == "int-123"
    assert result.model == MODEL
    assert result.status == "in_progress"
    assert result.created_at == 1793354400
    assert result.outputs == []
    assert result.error is None


def test_convert_interaction_falls_back_to_the_requested_model() -> None:
    result = convert_interaction_to_video_job(_interaction(model=None), fallback_model="requested")
    assert result.model == "requested"


def test_convert_interaction_reads_inline_video_data() -> None:
    interaction = _interaction("completed", steps=[_video_step(data=VIDEO_B64, mime_type="video/mp4")])
    result = convert_interaction_to_video_job(interaction, fallback_model=MODEL)

    assert result.done
    assert result.outputs == [VideoOutput(url=None, b64_data=VIDEO_B64, mime_type="video/mp4")]


def test_convert_interaction_reads_a_video_uri() -> None:
    uri = "https://generativelanguage.googleapis.com/v1beta/files/abc:download?alt=media"
    interaction = _interaction("completed", steps=[_video_step(uri=uri)])
    result = convert_interaction_to_video_job(interaction, fallback_model=MODEL)

    assert result.outputs == [VideoOutput(url=uri, b64_data=None, mime_type="video/mp4")]


def test_convert_interaction_reports_the_first_error() -> None:
    interaction = _interaction("failed", errors=[{"code": "SAFETY", "message": "blocked"}, {"message": "other"}])
    result = convert_interaction_to_video_job(interaction, fallback_model=MODEL)

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.message == "blocked"
    assert result.error.code == "SAFETY"


@pytest.mark.parametrize("status", ["cancelled", "incomplete", "requires_action", "budget_exceeded"])
def test_convert_interaction_treats_every_stopped_status_as_failed(status: str) -> None:
    result = convert_interaction_to_video_job(_interaction(status), fallback_model=MODEL)

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.message == f"Video generation ended with status {status}."


def test_convert_interaction_maps_queued() -> None:
    assert convert_interaction_to_video_job(_interaction("queued"), fallback_model=MODEL).status == "queued"


def test_convert_interaction_rejects_an_unknown_status() -> None:
    with pytest.raises(ProviderError, match="unknown interaction status"):
        convert_interaction_to_video_job(_interaction("exploded"), fallback_model=MODEL)


def test_convert_interaction_rejects_a_missing_id() -> None:
    with pytest.raises(ProviderError, match="without an id"):
        convert_interaction_to_video_job(_interaction(id=""), fallback_model=MODEL)


@pytest.mark.asyncio
async def test_avideo_generation_creates_a_background_interaction() -> None:
    provider, client = make_provider()
    client.aio.interactions.create = AsyncMock(return_value=_interaction())

    result = await provider.avideo_generation(MODEL, "a paper boat", size="1280x720", timeout=5)

    client.aio.interactions.create.assert_awaited_once_with(
        model=MODEL,
        input="a paper boat",
        background=True,
        response_format={"type": "video", "aspect_ratio": "16:9", "resolution": "720p"},
        timeout=5,
    )
    assert result.id == "int-123"
    assert result.status == "in_progress"


@pytest.mark.asyncio
async def test_avideo_generation_follows_the_clients_api_version() -> None:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        provider = GeminiProvider(api_key="test-key", http_options={"api_version": "v1"})
    client.aio.interactions.create = AsyncMock(return_value=_interaction())
    client.aio.interactions.get = AsyncMock(return_value=_interaction("completed"))

    await provider.avideo_generation(MODEL, "a paper boat")
    await provider.aretrieve_video("int-123")

    assert client.aio.interactions.create.await_args.kwargs["api_version"] == "v1"
    client.aio.interactions.get.assert_awaited_once_with("int-123", api_version="v1")


@pytest.mark.asyncio
async def test_aretrieve_video_gets_the_interaction() -> None:
    provider, client = make_provider()
    client.aio.interactions.get = AsyncMock(
        return_value=_interaction("completed", steps=[_video_step(data=VIDEO_B64, mime_type="video/mp4")])
    )

    result = await provider.aretrieve_video("int-123", timeout=5)

    client.aio.interactions.get.assert_awaited_once_with("int-123", timeout=5)
    assert result.done
    assert result.outputs[0].b64_data == VIDEO_B64


@pytest.mark.asyncio
async def test_adownload_video_decodes_inline_data_without_the_network() -> None:
    provider, client = make_provider()
    client.aio.files.download = AsyncMock()

    assert await provider.adownload_video(VideoOutput(b64_data=VIDEO_B64)) == b"video"
    client.aio.files.download.assert_not_called()


@pytest.mark.asyncio
async def test_adownload_video_uses_the_sdk_for_files_api_uris() -> None:
    provider, client = make_provider()
    client.aio.files.download = AsyncMock(return_value=b"video")
    uri = "https://generativelanguage.googleapis.com/v1beta/files/abc:download?alt=media"

    assert await provider.adownload_video(VideoOutput(url=uri)) == b"video"
    client.aio.files.download.assert_awaited_once_with(file=uri)


@pytest.mark.asyncio
async def test_adownload_video_fetches_other_urls_directly() -> None:
    provider, client = make_provider()
    client.aio.files.download = AsyncMock()

    with patch("any_llm.any_llm.AnyLLM._adownload_video", AsyncMock(return_value=b"video")) as base_download:
        assert await provider.adownload_video(VideoOutput(url="https://cdn.example/v.mp4")) == b"video"

    base_download.assert_awaited_once_with("https://cdn.example/v.mp4")
    client.aio.files.download.assert_not_called()
