import json
import re
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from together.types import VideoJob as TogetherVideoJob

from any_llm.exceptions import InvalidRequestError, ProviderError
from any_llm.providers.together.together import TogetherProvider
from any_llm.providers.together.utils import _convert_video_job
from any_llm.types.video import VideoGenerationParams, VideoJob


def make_video_job(**overrides: Any) -> TogetherVideoJob:
    """Build a Together VideoJob the lenient way the SDK parses API responses.

    ``construct`` skips validation, so a status missing from the SDK's Literal gets through
    the way it would from a live response.
    """
    fields: dict[str, Any] = {
        "id": "vid-123",
        "created_at": 1760000000.5,
        "model": "ByteDance/Seedance-2.0",
        "seconds": "5",
        "size": "1280x720",
        "status": "in_progress",
        "completed_at": None,
        "error": None,
        "object": "video",
        "outputs": None,
    }
    fields.update(overrides)
    return TogetherVideoJob.construct(**fields)


def make_provider() -> tuple[TogetherProvider, Mock]:
    with patch("any_llm.providers.together.together.together.AsyncTogether") as mock_together:
        mock_client = Mock()
        mock_together.return_value = mock_client
        provider = TogetherProvider(api_key="test-api-key")
    return provider, mock_client


def test_convert_video_job_maps_a_running_job() -> None:
    result = _convert_video_job(make_video_job())

    assert isinstance(result, VideoJob)
    assert result.id == "vid-123"
    assert result.model == "ByteDance/Seedance-2.0"
    assert result.status == "in_progress"
    assert result.created_at == 1760000000
    assert result.completed_at is None
    assert result.seconds == 5
    assert result.size == "1280x720"
    assert result.error is None
    assert result.outputs == []
    assert not result.done


def test_convert_video_job_maps_a_completed_job_to_one_url_output() -> None:
    together_job = make_video_job(
        status="completed",
        completed_at=1760000100.0,
        outputs={"cost": 16, "video_url": "https://cdn.together.ai/v.mp4"},
    )
    result = _convert_video_job(TogetherVideoJob.model_validate(together_job.model_dump()))

    assert result.status == "completed"
    assert result.done
    assert result.completed_at == 1760000100
    assert [output.url for output in result.outputs] == ["https://cdn.together.ai/v.mp4"]
    assert result.outputs[0].b64_data is None


def test_convert_video_job_maps_a_failed_job_to_an_error() -> None:
    together_job = make_video_job(status="failed", error={"message": "unsafe prompt", "code": "content_policy"})
    result = _convert_video_job(TogetherVideoJob.model_validate(together_job.model_dump()))

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.message == "unsafe prompt"
    assert result.error.code == "content_policy"
    assert result.outputs == []


@pytest.mark.parametrize(
    ("together_status", "status"),
    [("queued", "queued"), ("pending", "queued"), ("processing", "in_progress")],
)
def test_convert_video_job_maps_statuses_outside_the_sdk_literal(together_status: str, status: str) -> None:
    assert _convert_video_job(make_video_job(status=together_status)).status == status


def test_convert_video_job_rejects_an_unknown_status() -> None:
    with pytest.raises(ProviderError, match="unknown video status 'exploded' for job vid-123"):
        _convert_video_job(make_video_job(status="exploded"))


@pytest.mark.parametrize("seconds", ["", "five", None])
def test_convert_video_job_drops_unparseable_seconds(seconds: str | None) -> None:
    assert _convert_video_job(make_video_job(seconds=seconds)).seconds is None


def test_convert_video_job_drops_an_empty_size() -> None:
    assert _convert_video_job(make_video_job(size="")).size is None


def test_convert_video_params_sends_only_model_and_prompt_by_default() -> None:
    params = VideoGenerationParams(model_id="ByteDance/Seedance-2.0", prompt="penguins")
    assert TogetherProvider._convert_video_params(params) == {"model": "ByteDance/Seedance-2.0", "prompt": "penguins"}


def test_convert_video_params_maps_every_field_to_togethers_names() -> None:
    params = VideoGenerationParams(
        model_id="ByteDance/Seedance-2.0",
        prompt="penguins",
        seconds=5,
        size="1280x720",
        seed=7,
        negative_prompt="blurry",
        reference_images=["https://img/ref.png"],
        first_frame="https://img/first.png",
        last_frame="https://img/last.png",
    )
    assert TogetherProvider._convert_video_params(params, fps=24, steps=30) == {
        "model": "ByteDance/Seedance-2.0",
        "prompt": "penguins",
        "seconds": "5",
        "width": 1280,
        "height": 720,
        "seed": 7,
        "negative_prompt": "blurry",
        "media": {
            "reference_images": ["https://img/ref.png"],
            "frame_images": [
                {"input_image": "https://img/first.png", "frame": "first"},
                {"input_image": "https://img/last.png", "frame": "last"},
            ],
        },
        "fps": 24,
        "steps": 30,
    }


def test_convert_video_params_sends_resolution_and_ratio_as_arguments() -> None:
    params = VideoGenerationParams(
        model_id="ByteDance/Seedance-2.5", prompt="penguins", resolution="480p", aspect_ratio="16:9"
    )
    assert TogetherProvider._convert_video_params(params, extra_body={"output_quality": 30}) == {
        "model": "ByteDance/Seedance-2.5",
        "prompt": "penguins",
        "resolution": "480p",
        "ratio": "16:9",
        "extra_body": {"output_quality": 30},
    }


def test_convert_video_params_keeps_an_uppercase_resolution() -> None:
    params = VideoGenerationParams(model_id="Wan-AI/wan2.7-r2v", prompt="penguins", resolution="720P")
    assert TogetherProvider._convert_video_params(params) == {
        "model": "Wan-AI/wan2.7-r2v",
        "prompt": "penguins",
        "resolution": "720P",
    }


@pytest.mark.parametrize(
    ("resolution", "options", "expected"),
    [
        ("4k", {"resolution": "4k"}, {"resolution": "2160p"}),
        ("720p", {"resolution": "720P"}, {"resolution": "720p"}),
        ("720p", {"extra_body": {"resolution": "0720p"}}, {"resolution": "720p", "extra_body": {"resolution": "720p"}}),
        ("0720p", {"resolution": "720p"}, {"resolution": "720p"}),
        ("4K", {"extra_body": {"resolution": "4k"}}, {"resolution": "2160p", "extra_body": {"resolution": "2160p"}}),
        ("0720P", {"resolution": "720p"}, {"resolution": "720P"}),
        ("720P", {"extra_body": {"resolution": "720p"}}, {"resolution": "720P", "extra_body": {"resolution": "720P"}}),
    ],
)
def test_convert_video_params_accepts_a_resolution_spelled_another_way(
    resolution: str, options: dict[str, Any], expected: dict[str, Any]
) -> None:
    params = VideoGenerationParams(model_id="m", prompt="p", resolution=resolution)
    assert TogetherProvider._convert_video_params(params, **options) == {"model": "m", "prompt": "p", **expected}


def test_convert_video_params_rejects_size_combined_with_resolution() -> None:
    params = VideoGenerationParams(
        model_id="ByteDance/Seedance-2.5", prompt="penguins", size="854x480", resolution="480p"
    )
    with pytest.raises(InvalidRequestError, match="either size or resolution and aspect_ratio, not both"):
        TogetherProvider._convert_video_params(params)


def test_convert_video_params_sends_reference_audio_as_media_audio_inputs() -> None:
    params = VideoGenerationParams(
        model_id="ByteDance/Seedance-2.5", prompt="a rapper", reference_audios=["https://audio/beat.mp3"]
    )
    assert TogetherProvider._convert_video_params(params) == {
        "model": "ByteDance/Seedance-2.5",
        "prompt": "a rapper",
        "media": {"audio_inputs": ["https://audio/beat.mp3"]},
    }


@pytest.mark.parametrize("location", ["argument", "extra_body"])
def test_convert_video_params_merges_reference_audio_into_the_callers_media(location: str) -> None:
    params = VideoGenerationParams(
        model_id="ByteDance/Seedance-2.5", prompt="a rapper", reference_audios=["https://audio/beat.mp3"]
    )
    media = {"reference_videos": [{"video": "https://v/ref.mp4"}]}
    options: dict[str, Any] = {"media": media} if location == "argument" else {"extra_body": {"media": media}}
    converted = TogetherProvider._convert_video_params(params, **options)
    assert converted["media"] == {
        "reference_videos": [{"video": "https://v/ref.mp4"}],
        "audio_inputs": ["https://audio/beat.mp3"],
    }
    assert "extra_body" not in converted
    # The caller's dict is left as it was.
    assert media == {"reference_videos": [{"video": "https://v/ref.mp4"}]}


def test_convert_video_params_lets_media_as_an_argument_win_over_media_in_extra_body() -> None:
    params = VideoGenerationParams(model_id="m", prompt="p")
    converted = TogetherProvider._convert_video_params(
        params,
        media={"source_video": "https://v/new.mp4"},
        extra_body={"media": {"source_video": "https://v/old.mp4", "frame_videos": [{"video": "https://v/f.mp4"}]}},
    )
    assert converted["media"] == {"source_video": "https://v/new.mp4", "frame_videos": [{"video": "https://v/f.mp4"}]}


def test_convert_video_params_sends_reference_videos_as_media_reference_videos() -> None:
    params = VideoGenerationParams(
        model_id="ByteDance/Seedance-2.0",
        prompt="a dancer",
        reference_videos=["https://v/one.mp4", "https://v/two.mp4"],
    )
    assert TogetherProvider._convert_video_params(params) == {
        "model": "ByteDance/Seedance-2.0",
        "prompt": "a dancer",
        "media": {"reference_videos": [{"video": "https://v/one.mp4"}, {"video": "https://v/two.mp4"}]},
    }


def test_convert_video_params_accepts_media_that_repeats_reference_videos_and_keeps_audio() -> None:
    params = VideoGenerationParams(
        model_id="ByteDance/Seedance-2.0",
        prompt="a dancer",
        reference_audios=["https://audio/beat.mp3"],
        reference_videos=["https://v/mine.mp4"],
    )
    media = {"reference_videos": [{"video": "https://v/mine.mp4"}]}
    converted = TogetherProvider._convert_video_params(params, media=media)
    assert converted["media"] == {
        "reference_videos": [{"video": "https://v/mine.mp4"}],
        "audio_inputs": ["https://audio/beat.mp3"],
    }
    assert media == {"reference_videos": [{"video": "https://v/mine.mp4"}]}


@pytest.mark.parametrize("generate_audio", [True, False])
def test_convert_video_params_sends_generate_audio_as_an_argument(generate_audio: bool) -> None:
    params = VideoGenerationParams(model_id="ByteDance/Seedance-2.0", prompt="a dancer", generate_audio=generate_audio)
    assert TogetherProvider._convert_video_params(params) == {
        "model": "ByteDance/Seedance-2.0",
        "prompt": "a dancer",
        "generate_audio": generate_audio,
    }


@pytest.mark.parametrize(
    ("shared", "options", "message"),
    [
        ({"seconds": 5}, {"seconds": "10"}, "seconds=5 conflicts with seconds='10'"),
        ({"size": "1280x720"}, {"width": 1920}, "size='1280x720' conflicts with width=1920"),
        ({"seed": 7}, {"seed": 8}, "seed=7 conflicts with seed=8"),
        ({"negative_prompt": "blur"}, {"negative_prompt": "rain"}, "negative_prompt='blur' conflicts"),
        ({"reference_images": ["a"]}, {"media": {"reference_images": ["b"]}}, "reference_images=['a'] conflicts"),
        ({"resolution": "720p"}, {"resolution": "1080p"}, "resolution='720p' conflicts"),
        ({"resolution": "720p"}, {"extra_body": {"resolution": "1080p"}}, "resolution='720p' conflicts"),
        ({"resolution": "720p"}, {"extra_body": {"resolution": "0480p"}}, "resolution='720p' conflicts"),
        ({"resolution": "720p"}, {"resolution": "hd"}, "resolution='720p' conflicts"),
        ({"resolution": "720p"}, {"resolution": 720}, "resolution='720p' conflicts"),
        ({"aspect_ratio": "16:9"}, {"extra_body": {"ratio": "9:16"}}, "aspect_ratio='16:9' conflicts with ratio"),
        ({"generate_audio": False}, {"generate_audio": True}, "generate_audio=False conflicts"),
        ({"generate_audio": False}, {"extra_body": {"generate_audio": True}}, "generate_audio=False conflicts"),
        ({"size": "1280x720"}, {"extra_body": {"width": 1920}}, "size='1280x720' conflicts with width=1920"),
        ({"seconds": 5}, {"extra_body": {"seconds": "10"}}, "seconds=5 conflicts with seconds='10'"),
        (
            {"first_frame": "a.png"},
            {"media": {"frame_images": [{"input_image": "b.png", "frame": "first"}]}},
            "first_frame='a.png' conflicts with frame_images[first]=['b.png']",
        ),
        (
            {"first_frame": "a.png"},
            {
                "media": {
                    "frame_images": [
                        {"input_image": "b.png", "frame": "first"},
                        {"input_image": "a.png", "frame": "first"},
                    ]
                }
            },
            "first_frame='a.png' conflicts with frame_images[first]=['b.png', 'a.png']",
        ),
        (
            {"last_frame": "a.png"},
            {"extra_body": {"media": {"frame_images": [{"input_image": "b.png", "frame": "last"}]}}},
            "last_frame='a.png' conflicts with frame_images[last]=['b.png']",
        ),
        ({}, {"extra_body": {"prompt": "a goose"}}, "prompt='a dancer' conflicts with prompt='a goose'"),
        (
            {"reference_videos": ["v.mp4"]},
            {"extra_body": {"media": {"reference_videos": [{"video": "w.mp4"}]}}},
            "reference_videos=['v.mp4'] conflicts with reference_videos=[{'video': 'w.mp4'}]",
        ),
        (
            {"reference_audios": ["a"]},
            {"media": {"audio_inputs": ["b"]}},
            "reference_audios=['a'] conflicts with audio_inputs=['b']",
        ),
        (
            {},
            {"extra_body": {"model": "Kwai/Kling-2.1"}},
            "model_id='ByteDance/Seedance-2.0' conflicts with model='Kwai/Kling-2.1'",
        ),
        (
            {"reference_images": ["a"]},
            {"media": {"reference_images": ["a"]}, "extra_body": {"media": {"reference_images": ["b"]}}},
            "reference_images=['a'] conflicts with reference_images=['b']",
        ),
        (
            {"first_frame": "a.png"},
            {
                "media": {"frame_images": [{"input_image": "a.png", "frame": "first"}]},
                "extra_body": {"media": {"frame_images": [{"input_image": "b.png", "frame": "first"}]}},
            },
            "first_frame='a.png' conflicts with frame_images[first]=['b.png']",
        ),
    ],
)
def test_convert_video_params_rejects_an_option_that_contradicts_a_shared_field(
    shared: dict[str, Any], options: dict[str, Any], message: str
) -> None:
    params = VideoGenerationParams(model_id="ByteDance/Seedance-2.0", prompt="a dancer", **shared)
    with pytest.raises(InvalidRequestError, match=re.escape(message)):
        TogetherProvider._convert_video_params(params, **options)


def test_convert_video_params_accepts_options_that_agree_with_the_shared_fields() -> None:
    params = VideoGenerationParams(
        model_id="ByteDance/Seedance-2.0", prompt="a dancer", size="1280x720", generate_audio=False
    )
    extra_body = {"model": "ByteDance/Seedance-2.0", "generate_audio": False, "fps": 24}
    converted = TogetherProvider._convert_video_params(params, width=1280, extra_body=extra_body)
    assert converted == {
        "model": "ByteDance/Seedance-2.0",
        "prompt": "a dancer",
        "width": 1280,
        "height": 720,
        "generate_audio": False,
        "extra_body": {"model": "ByteDance/Seedance-2.0", "generate_audio": False, "fps": 24},
    }
    assert extra_body == {"model": "ByteDance/Seedance-2.0", "generate_audio": False, "fps": 24}


def test_convert_video_params_keeps_one_frame_image_when_the_caller_repeats_last_frame() -> None:
    params = VideoGenerationParams(model_id="m", prompt="p", last_frame="https://img/last.png")
    converted = TogetherProvider._convert_video_params(
        params, media={"frame_images": [{"input_image": "https://img/last.png", "frame": "last"}]}
    )
    assert converted["media"] == {"frame_images": [{"input_image": "https://img/last.png", "frame": "last"}]}


def test_convert_video_params_merges_extra_body_frame_images_with_first_frame() -> None:
    params = VideoGenerationParams(model_id="m", prompt="p", first_frame="https://img/first.png")
    extra_body = {"media": {"frame_images": [{"input_image": "https://img/mid.png", "frame": 0.5}]}, "fps": 24}
    converted = TogetherProvider._convert_video_params(params, extra_body=extra_body)
    assert converted["media"]["frame_images"] == [
        {"input_image": "https://img/mid.png", "frame": 0.5},
        {"input_image": "https://img/first.png", "frame": "first"},
    ]
    assert converted["extra_body"] == {"fps": 24}


def test_convert_video_params_merges_caller_frame_images_with_first_and_last_frames() -> None:
    params = VideoGenerationParams(
        model_id="m", prompt="p", first_frame="https://img/first.png", last_frame="https://img/last.png"
    )
    converted = TogetherProvider._convert_video_params(
        params, media={"frame_images": [{"input_image": "https://img/mid.png", "frame": 0.5}]}
    )
    assert converted["media"]["frame_images"] == [
        {"input_image": "https://img/mid.png", "frame": 0.5},
        {"input_image": "https://img/first.png", "frame": "first"},
        {"input_image": "https://img/last.png", "frame": "last"},
    ]


def test_convert_video_params_rejects_a_malformed_size() -> None:
    params = VideoGenerationParams(model_id="m", prompt="p", size="720p")
    with pytest.raises(InvalidRequestError, match="WIDTHxHEIGHT"):
        TogetherProvider._convert_video_params(params)


@pytest.mark.asyncio
async def test_avideo_generation_creates_a_job_and_converts_it() -> None:
    provider, mock_client = make_provider()
    mock_client.videos.create = AsyncMock(return_value=make_video_job())

    result = await provider.avideo_generation("ByteDance/Seedance-2.0", "penguins", seconds=5, size="1280x720", fps=24)

    mock_client.videos.create.assert_awaited_once_with(
        model="ByteDance/Seedance-2.0", prompt="penguins", seconds="5", width=1280, height=720, fps=24
    )
    assert isinstance(result, VideoJob)
    assert result.id == "vid-123"
    assert result.status == "in_progress"


@pytest.mark.asyncio
async def test_avideo_generation_rejects_a_width_that_contradicts_size_before_creating() -> None:
    provider, mock_client = make_provider()
    mock_client.videos.create = AsyncMock(return_value=make_video_job())

    with pytest.raises(InvalidRequestError, match="size='1280x720' conflicts with width=1920"):
        await provider.avideo_generation("ByteDance/Seedance-2.0", "penguins", size="1280x720", width=1920)

    mock_client.videos.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_aretrieve_video_fetches_the_job_by_id() -> None:
    provider, mock_client = make_provider()
    mock_client.videos.retrieve = AsyncMock(
        return_value=make_video_job(status="completed", outputs={"cost": 16, "video_url": "https://cdn/v.mp4"})
    )

    result = await provider.aretrieve_video("vid-123", timeout=5)

    mock_client.videos.retrieve.assert_awaited_once_with("vid-123", timeout=5)
    assert result.done
    assert result.outputs[0].url == "https://cdn/v.mp4"


@pytest.mark.asyncio
async def test_apoll_video_returns_once_together_reports_completion() -> None:
    provider, mock_client = make_provider()
    mock_client.videos.retrieve = AsyncMock(
        side_effect=[
            make_video_job(),
            make_video_job(status="completed", outputs={"cost": 16, "video_url": "https://cdn/v.mp4"}),
        ]
    )

    with patch("any_llm.any_llm.asyncio.sleep", AsyncMock()):
        result = await provider.apoll_video("vid-123", poll_interval=0)

    assert mock_client.videos.retrieve.await_count == 2
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_avideo_generation_sends_media_in_the_request_body() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=make_video_job(status="queued").model_dump(mode="json"))

    provider = TogetherProvider(
        api_key="test-api-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    params = VideoGenerationParams(
        model_id="ByteDance/Seedance-2.5",
        prompt="penguins",
        resolution="720p",
        aspect_ratio="16:9",
        generate_audio=True,
        first_frame="https://img/first.png",
        reference_images=["https://img/ref.png"],
    )

    job = await provider._avideo_generation(params, fps=24)

    assert job.status == "queued"
    assert requests[0].url == "https://api.together.ai/v2/videos"
    assert json.loads(requests[0].content) == {
        "model": "ByteDance/Seedance-2.5",
        "prompt": "penguins",
        "resolution": "720p",
        "ratio": "16:9",
        "generate_audio": True,
        "media": {
            "reference_images": ["https://img/ref.png"],
            "frame_images": [{"input_image": "https://img/first.png", "frame": "first"}],
        },
        "fps": 24,
    }
