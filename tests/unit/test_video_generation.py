from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from any_llm import AnyLLM
from any_llm.api import (
    adownload_video,
    apoll_video,
    aretrieve_video,
    avideo_generation,
    download_video,
    poll_video,
    retrieve_video,
    video_generation,
)
from any_llm.constants import LLMProvider
from any_llm.exceptions import InvalidRequestError
from any_llm.types.video import VideoGenerationParams, VideoJob, VideoOutput


def _job(status: str = "queued", **overrides: object) -> VideoJob:
    fields: dict[str, object] = {"id": "vid-1", "model": "some-model", "status": status}
    fields.update(overrides)
    return VideoJob.model_validate(fields)


def _base(supports: bool = True) -> Mock:
    base = Mock(spec=AnyLLM)
    base.SUPPORTS_VIDEO_GENERATION = supports
    return base


@pytest.mark.asyncio
async def test_avideo_generation_with_api_config() -> None:
    provider = Mock()
    provider.avideo_generation = AsyncMock(return_value=_job())

    with patch("any_llm.any_llm.AnyLLM.create", return_value=provider) as mock_create:
        result = await avideo_generation(
            "together:ByteDance/Seedance-2.0",
            prompt="A cat",
            api_key="test_key",
            api_base="https://test.example.com",
        )

    assert mock_create.call_args[0][0] == LLMProvider.TOGETHER
    assert mock_create.call_args[1]["api_key"] == "test_key"
    assert mock_create.call_args[1]["api_base"] == "https://test.example.com"
    assert provider.avideo_generation.call_args[0] == ("ByteDance/Seedance-2.0", "A cat")
    assert result.id == "vid-1"


@pytest.mark.asyncio
async def test_avideo_generation_with_explicit_provider_passes_all_params() -> None:
    provider = Mock()
    provider.avideo_generation = AsyncMock(return_value=_job())

    with patch("any_llm.any_llm.AnyLLM.create", return_value=provider) as mock_create:
        await avideo_generation(
            "ByteDance/Seedance-2.0",
            prompt="A dog",
            provider="together",
            seconds=5,
            size="1280x720",
            seed=7,
            negative_prompt="blurry",
            reference_images=["https://img/1.png"],
            first_frame="https://img/first.png",
            last_frame="https://img/last.png",
            fps=24,
        )

    assert mock_create.call_args[0][0] == LLMProvider.TOGETHER
    assert provider.avideo_generation.call_args[0] == ("ByteDance/Seedance-2.0", "A dog")
    assert provider.avideo_generation.call_args[1] == {
        "seconds": 5,
        "size": "1280x720",
        "seed": 7,
        "negative_prompt": "blurry",
        "reference_images": ["https://img/1.png"],
        "first_frame": "https://img/first.png",
        "last_frame": "https://img/last.png",
        "fps": 24,
    }


def test_sync_video_generation_dispatches() -> None:
    provider = Mock()
    provider.video_generation = Mock(return_value=_job())

    with patch("any_llm.any_llm.AnyLLM.create", return_value=provider):
        result = video_generation("together:ByteDance/Seedance-2.0", prompt="A fish", seconds=5)

    assert provider.video_generation.call_args[0] == ("ByteDance/Seedance-2.0", "A fish")
    assert provider.video_generation.call_args[1]["seconds"] == 5
    assert result.id == "vid-1"


def test_sync_video_generation_with_explicit_provider_keeps_the_model_whole() -> None:
    provider = Mock()
    provider.video_generation = Mock(return_value=_job())

    with patch("any_llm.any_llm.AnyLLM.create", return_value=provider) as mock_create:
        video_generation("ByteDance/Seedance-2.0", prompt="A fish", provider="together", api_key="k")

    assert mock_create.call_args[0][0] == LLMProvider.TOGETHER
    assert mock_create.call_args[1]["api_key"] == "k"
    assert provider.video_generation.call_args[0] == ("ByteDance/Seedance-2.0", "A fish")


@pytest.mark.asyncio
async def test_base_avideo_generation_builds_params_and_forwards_kwargs() -> None:
    base = _base()
    base._avideo_generation = AsyncMock(return_value=_job())

    result = await AnyLLM.avideo_generation(base, "some-model", "A cat", seconds=5, size="1280x720", fps=24)

    params = base._avideo_generation.call_args[0][0]
    assert isinstance(params, VideoGenerationParams)
    assert params.model_id == "some-model"
    assert params.prompt == "A cat"
    assert params.seconds == 5
    assert params.size == "1280x720"
    assert base._avideo_generation.call_args[1] == {"fps": 24}
    assert result.status == "queued"


@pytest.mark.asyncio
async def test_avideo_generation_unsupported_provider_raises_not_implemented() -> None:
    base = _base(supports=False)
    with pytest.raises(NotImplementedError, match="doesn't support video generation"):
        await AnyLLM.avideo_generation(base, "some-model", "A cat")
    base._avideo_generation.assert_not_called()


@pytest.mark.asyncio
async def test_aretrieve_video_unsupported_provider_raises_not_implemented() -> None:
    base = _base(supports=False)
    with pytest.raises(NotImplementedError, match="doesn't support video generation"):
        await AnyLLM.aretrieve_video(base, "vid-1")
    base._aretrieve_video.assert_not_called()


@pytest.mark.asyncio
async def test_aretrieve_video_forwards_to_the_provider() -> None:
    base = _base()
    base._aretrieve_video = AsyncMock(return_value=_job("completed"))

    result = await AnyLLM.aretrieve_video(base, "vid-1", extra="x")

    base._aretrieve_video.assert_awaited_once_with("vid-1", extra="x")
    assert result.done


@pytest.mark.asyncio
async def test_base_methods_raise_when_a_supporting_provider_has_no_implementation() -> None:
    base = _base()
    with pytest.raises(NotImplementedError, match="Subclasses must implement _avideo_generation"):
        await AnyLLM._avideo_generation(base, VideoGenerationParams(model_id="some-model", prompt="A cat"))
    with pytest.raises(NotImplementedError, match="Subclasses must implement _aretrieve_video"):
        await AnyLLM._aretrieve_video(base, "vid-1")


@pytest.mark.asyncio
async def test_apoll_video_returns_the_first_finished_job() -> None:
    base = _base()
    base.aretrieve_video = AsyncMock(side_effect=[_job("queued"), _job("in_progress"), _job("completed")])

    with patch("any_llm.any_llm.asyncio.sleep", AsyncMock()) as mock_sleep:
        result = await AnyLLM.apoll_video(base, "vid-1", poll_interval=2.5, extra="x")

    assert result.status == "completed"
    assert base.aretrieve_video.await_count == 3
    assert base.aretrieve_video.await_args_list[0].kwargs == {"extra": "x"}
    assert mock_sleep.await_count == 2
    assert mock_sleep.await_args_list[-1].args == (2.5,)


@pytest.mark.asyncio
async def test_apoll_video_returns_a_failed_job_instead_of_raising() -> None:
    base = _base()
    base.aretrieve_video = AsyncMock(return_value=_job("failed", error={"message": "boom"}))

    with patch("any_llm.any_llm.asyncio.sleep", AsyncMock()) as mock_sleep:
        result = await AnyLLM.apoll_video(base, "vid-1")

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.message == "boom"
    mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_apoll_video_times_out_with_the_last_status() -> None:
    base = _base()
    base.aretrieve_video = AsyncMock(return_value=_job("in_progress"))

    with (
        patch("any_llm.any_llm.asyncio.sleep", AsyncMock()),
        patch("any_llm.any_llm.time", Mock(monotonic=Mock(side_effect=[0.0, 1.0, 10.0]))),
        pytest.raises(TimeoutError, match="vid-1 is still in_progress after 10 seconds"),
    ):
        await AnyLLM.apoll_video(base, "vid-1", poll_interval=1.0, timeout=10)

    assert base.aretrieve_video.await_count == 2


@pytest.mark.asyncio
async def test_apoll_video_without_timeout_never_checks_the_clock() -> None:
    base = _base()
    base.aretrieve_video = AsyncMock(side_effect=[_job("queued"), _job("completed")])

    with (
        patch("any_llm.any_llm.asyncio.sleep", AsyncMock()),
        patch("any_llm.any_llm.time", Mock(monotonic=Mock(side_effect=AssertionError("clock read")))),
    ):
        result = await AnyLLM.apoll_video(base, "vid-1", timeout=None)

    assert result.done


@pytest.mark.asyncio
async def test_adownload_video_decodes_inline_data() -> None:
    base = _base()
    result = await AnyLLM.adownload_video(base, VideoOutput(b64_data="dmlkZW8="))
    assert result == b"video"
    base._adownload_video.assert_not_called()


@pytest.mark.asyncio
async def test_adownload_video_fetches_a_url() -> None:
    base = _base()
    base._adownload_video = AsyncMock(return_value=b"video")

    result = await AnyLLM.adownload_video(base, VideoOutput(url="https://cdn.example/v.mp4"), timeout=3)

    base._adownload_video.assert_awaited_once_with("https://cdn.example/v.mp4", timeout=3)
    assert result == b"video"


@pytest.mark.asyncio
async def test_adownload_video_rejects_an_empty_output() -> None:
    base = _base()
    with pytest.raises(InvalidRequestError, match="neither a URL nor inline data"):
        await AnyLLM.adownload_video(base, VideoOutput())


@pytest.mark.asyncio
async def test_base_adownload_video_gets_the_url_and_follows_redirects() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/old.mp4":
            return httpx.Response(302, headers={"location": "https://cdn.example/new.mp4"})
        return httpx.Response(200, content=b"video")

    result = await AnyLLM._adownload_video(
        _base(), "https://cdn.example/old.mp4", transport=httpx.MockTransport(handler)
    )

    assert result == b"video"
    assert [request.url.path for request in requests] == ["/old.mp4", "/new.mp4"]


@pytest.mark.asyncio
async def test_base_adownload_video_raises_on_http_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(httpx.HTTPStatusError):
        await AnyLLM._adownload_video(_base(), "https://cdn.example/v.mp4", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_async_provider_only_entry_points_dispatch() -> None:
    provider = Mock()
    provider.aretrieve_video = AsyncMock(return_value=_job("in_progress"))
    provider.apoll_video = AsyncMock(return_value=_job("completed"))
    provider.adownload_video = AsyncMock(return_value=b"video")
    output = VideoOutput(url="https://cdn.example/v.mp4")

    with patch("any_llm.any_llm.AnyLLM.create", return_value=provider) as mock_create:
        retrieved = await aretrieve_video("together", "vid-1", api_key="k")
        polled = await apoll_video("together", "vid-1", poll_interval=1.0, timeout=None)
        downloaded = await adownload_video("together", output)

    assert all(call.args[0] == LLMProvider.TOGETHER for call in mock_create.call_args_list)
    assert mock_create.call_args_list[0].kwargs["api_key"] == "k"
    provider.aretrieve_video.assert_awaited_once_with("vid-1")
    provider.apoll_video.assert_awaited_once_with("vid-1", poll_interval=1.0, timeout=None)
    provider.adownload_video.assert_awaited_once_with(output)
    assert retrieved.status == "in_progress"
    assert polled.status == "completed"
    assert downloaded == b"video"


def test_sync_provider_only_entry_points_dispatch() -> None:
    provider = Mock()
    provider.retrieve_video = Mock(return_value=_job("in_progress"))
    provider.poll_video = Mock(return_value=_job("completed"))
    provider.download_video = Mock(return_value=b"video")
    output = VideoOutput(url="https://cdn.example/v.mp4")

    with patch("any_llm.any_llm.AnyLLM.create", return_value=provider):
        retrieved = retrieve_video("together", "vid-1")
        polled = poll_video("together", "vid-1", poll_interval=1.0, timeout=30)
        downloaded = download_video("together", output)

    provider.retrieve_video.assert_called_once_with("vid-1")
    provider.poll_video.assert_called_once_with("vid-1", poll_interval=1.0, timeout=30)
    provider.download_video.assert_called_once_with(output)
    assert retrieved.status == "in_progress"
    assert polled.status == "completed"
    assert downloaded == b"video"


def test_sync_base_methods_run_their_async_twins() -> None:
    base = _base()
    base.avideo_generation = AsyncMock(return_value=_job())
    base.aretrieve_video = AsyncMock(return_value=_job("in_progress"))
    base.apoll_video = AsyncMock(return_value=_job("completed"))
    base.adownload_video = AsyncMock(return_value=b"video")
    output = VideoOutput(b64_data="dmlkZW8=")

    assert AnyLLM.video_generation(base, "some-model", "A cat", seconds=5).status == "queued"
    assert AnyLLM.retrieve_video(base, "vid-1").status == "in_progress"
    assert AnyLLM.poll_video(base, "vid-1", poll_interval=1.0, timeout=None).status == "completed"
    assert AnyLLM.download_video(base, output) == b"video"

    base.avideo_generation.assert_awaited_once_with("some-model", "A cat", seconds=5)
    base.aretrieve_video.assert_awaited_once_with("vid-1")
    base.apoll_video.assert_awaited_once_with("vid-1", poll_interval=1.0, timeout=None)
    base.adownload_video.assert_awaited_once_with(output)


def test_provider_metadata_reports_video_generation() -> None:
    for provider_enum in LLMProvider:
        try:
            cls = AnyLLM.get_provider_class(provider_enum)
        except ImportError:
            continue
        assert cls.get_provider_metadata().video_generation is cls.SUPPORTS_VIDEO_GENERATION


def test_supports_video_generation_only_on_expected_providers() -> None:
    expected_supported: set[LLMProvider] = set()

    for provider_enum in expected_supported:
        cls = AnyLLM.get_provider_class(provider_enum)
        assert cls.SUPPORTS_VIDEO_GENERATION is True, f"{provider_enum.value} should support video generation"

    for provider_enum in LLMProvider:
        if provider_enum in expected_supported:
            continue
        try:
            cls = AnyLLM.get_provider_class(provider_enum)
        except ImportError:
            continue
        assert cls.SUPPORTS_VIDEO_GENERATION is False, f"{provider_enum.value} should not support video generation"
