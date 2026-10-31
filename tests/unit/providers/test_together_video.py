from typing import Any
from unittest.mock import AsyncMock, Mock, patch

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
        "reference_images": ["https://img/ref.png"],
        "frame_images": [
            {"input_image": "https://img/first.png", "frame": "first"},
            {"input_image": "https://img/last.png", "frame": "last"},
        ],
        "fps": 24,
        "steps": 30,
    }


def test_convert_video_params_merges_caller_frame_images_with_first_and_last_frames() -> None:
    params = VideoGenerationParams(model_id="m", prompt="p", last_frame="https://img/last.png")
    converted = TogetherProvider._convert_video_params(
        params, frame_images=[{"input_image": "https://img/mid.png", "frame": 0.5}]
    )
    assert converted["frame_images"] == [
        {"input_image": "https://img/mid.png", "frame": 0.5},
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
