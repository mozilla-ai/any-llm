import pytest

from any_llm.exceptions import InvalidRequestError
from any_llm.types.video import VideoError, VideoGenerationParams, VideoJob, VideoOutput


def test_video_job_defaults_to_no_outputs() -> None:
    job = VideoJob(id="vid-1", model="m", status="queued")
    assert job.outputs == []
    assert job.error is None
    assert job.progress is None


def test_video_job_reports_no_charge_by_default() -> None:
    job = VideoJob(id="vid-1", model="m", status="completed")
    assert job.cost is None
    assert job.tokens is None


def test_video_job_carries_the_provider_charge() -> None:
    job = VideoJob(id="vid-1", model="m", status="completed", cost=0.38991, tokens=50638)
    assert job.cost == 0.38991
    assert job.tokens == 50638


@pytest.mark.parametrize(
    ("status", "done"),
    [("queued", False), ("in_progress", False), ("completed", True), ("failed", True)],
)
def test_video_job_done_only_on_terminal_statuses(status: str, done: bool) -> None:
    job = VideoJob.model_validate({"id": "vid-1", "model": "m", "status": status})
    assert job.done is done


def test_video_job_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="status"):
        VideoJob.model_validate({"id": "vid-1", "model": "m", "status": "pending"})


def test_video_job_carries_outputs_and_error() -> None:
    job = VideoJob(
        id="vid-1",
        model="m",
        status="failed",
        error=VideoError(message="boom", code="E1"),
        outputs=[VideoOutput(url="https://cdn.example/v.mp4")],
    )
    assert job.error is not None
    assert job.error.message == "boom"
    assert job.outputs[0].mime_type == "video/mp4"
    assert job.outputs[0].b64_data is None


def test_video_generation_params_to_api_kwargs_excludes_none_and_model() -> None:
    params = VideoGenerationParams(model_id="m", prompt="a cat")
    assert params.to_api_kwargs() == {"prompt": "a cat"}


def test_video_generation_params_to_api_kwargs_includes_set_values() -> None:
    params = VideoGenerationParams(
        model_id="m",
        prompt="a cat",
        seconds=5,
        size="1280x720",
        seed=7,
        negative_prompt="blurry",
        reference_images=["https://img/1.png"],
        reference_videos=["https://video/1.mp4"],
        first_frame="https://img/first.png",
        last_frame="https://img/last.png",
        generate_audio=False,
    )
    assert params.to_api_kwargs() == {
        "prompt": "a cat",
        "seconds": 5,
        "size": "1280x720",
        "seed": 7,
        "negative_prompt": "blurry",
        "reference_images": ["https://img/1.png"],
        "reference_videos": ["https://video/1.mp4"],
        "first_frame": "https://img/first.png",
        "last_frame": "https://img/last.png",
        "generate_audio": False,
    }


def test_video_generation_params_rejects_extra_fields() -> None:
    with pytest.raises(ValueError, match="extra"):
        VideoGenerationParams(model_id="m", prompt="a cat", bogus="value")  # type: ignore[call-arg]


def test_parse_size_returns_none_without_size() -> None:
    assert VideoGenerationParams(model_id="m", prompt="a cat").parse_size() is None


@pytest.mark.parametrize("size", ["1280x720", "1280X720"])
def test_parse_size_splits_width_and_height(size: str) -> None:
    assert VideoGenerationParams(model_id="m", prompt="a cat", size=size).parse_size() == (1280, 720)


@pytest.mark.parametrize("size", ["720p", "1280", "x720", "1280x", "ax720", "1280x720x1", "1280x0", "0x720", "²x720"])
def test_parse_size_rejects_malformed_sizes(size: str) -> None:
    with pytest.raises(InvalidRequestError, match="WIDTHxHEIGHT"):
        VideoGenerationParams(model_id="m", prompt="a cat", size=size).parse_size()


def test_merge_provider_options_lays_options_over_the_shared_fields() -> None:
    params = VideoGenerationParams(model_id="m", prompt="p", seconds=5)
    merged = params.merge_provider_options({"duration": ("seconds", 5)}, {"fps": 24})
    assert merged == {"duration": 5, "fps": 24}


def test_merge_provider_options_accepts_an_option_that_agrees_with_a_shared_field() -> None:
    params = VideoGenerationParams(model_id="m", prompt="p", seconds=5)
    assert params.merge_provider_options({"duration": ("seconds", 5)}, {"duration": 5}) == {"duration": 5}


def test_merge_provider_options_rejects_an_option_that_contradicts_a_shared_field() -> None:
    params = VideoGenerationParams(model_id="m", prompt="p", size="1280x720")
    with pytest.raises(InvalidRequestError, match="size='1280x720' conflicts with width=1920; pass only one of them"):
        params.merge_provider_options({"width": ("size", 1280), "height": ("size", 720)}, {"width": 1920})


def test_check_provider_options_accepts_options_that_agree_or_set_other_fields() -> None:
    params = VideoGenerationParams(model_id="m", prompt="p", seconds=5)
    params.check_provider_options({"duration": ("seconds", 5)}, {"duration": 5, "fps": 24})


def test_check_provider_options_rejects_an_option_that_contradicts_a_shared_field() -> None:
    params = VideoGenerationParams(model_id="m", prompt="p", seconds=5)
    with pytest.raises(InvalidRequestError, match="seconds=5 conflicts with duration=10"):
        params.check_provider_options({"duration": ("seconds", 5)}, {"duration": 10})
