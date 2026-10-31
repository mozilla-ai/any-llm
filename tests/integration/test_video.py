import base64
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.video import VideoJob
from tests.constants import EXPECTED_PROVIDERS

# Every provider bills per clip, so one short, low-resolution clip per run. Gemini takes no
# length and renders 360p as its smallest size; Together takes both.
_PROMPT = "A paper boat drifting across a puddle, overcast light, static camera."
_FIRST_FRAME_PROMPT = "The logo slowly rotates in place while soft light sweeps across it."
_GENERATION_KWARGS: dict[LLMProvider, dict[str, Any]] = {
    LLMProvider.GEMINI: {"size": "640x360"},
    LLMProvider.TOGETHER: {"seconds": 4, "size": "864x496"},
}
_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_HOSTED_LOGO_URL = "https://raw.githubusercontent.com/mozilla-ai/any-llm/main/tests/assets/any-llm-logo.png"


def _first_frame(provider: LLMProvider) -> str:
    """The logo as the frame image each provider can take.

    Together's ``frame_images`` take only a hosted URL, while Gemini's Interactions API takes
    inline image data, so the same picture goes in through two doors.
    """
    if provider is LLMProvider.TOGETHER:
        return _HOSTED_LOGO_URL
    encoded = base64.b64encode((_ASSETS_DIR / "any-llm-logo.png").read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


async def _start_video_job(
    provider: LLMProvider,
    video_provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
    prompt: str,
    **kwargs: Any,
) -> tuple[AnyLLM, VideoJob]:
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_VIDEO_GENERATION:
            pytest.skip(f"{provider.value} does not support video generation, skipping")

        model_id = video_provider_model_map[provider]
        job = await llm.avideo_generation(model_id, prompt, **_GENERATION_KWARGS.get(provider, {}), **kwargs)
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} connection failed, skipping")

    assert isinstance(job, VideoJob)
    assert job.id
    assert job.model
    return llm, job


async def _finish_video_job(llm: AnyLLM, job: VideoJob) -> None:
    finished = await llm.apoll_video(job.id, poll_interval=10, timeout=900)
    assert finished.status == "completed", finished.error
    assert finished.outputs

    video = await llm.adownload_video(finished.outputs[0])
    assert len(video) > 1000


@pytest.mark.asyncio
async def test_video_generation_providers_async(
    provider: LLMProvider,
    video_provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Every video provider starts a job, reports it by id, and finishes it with a downloadable video."""
    llm, job = await _start_video_job(provider, video_provider_model_map, provider_client_config, _PROMPT)

    retrieved = await llm.aretrieve_video(job.id)
    assert retrieved.id == job.id

    await _finish_video_job(llm, job)


@pytest.mark.asyncio
async def test_video_generation_from_a_first_frame_async(
    provider: LLMProvider,
    video_provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Every video provider accepts a first frame and animates it into a downloadable video."""
    llm, job = await _start_video_job(
        provider,
        video_provider_model_map,
        provider_client_config,
        _FIRST_FRAME_PROMPT,
        first_frame=_first_frame(provider),
    )

    await _finish_video_job(llm, job)
