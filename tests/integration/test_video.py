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
_GENERATION_KWARGS: dict[LLMProvider, dict[str, Any]] = {
    LLMProvider.GEMINI: {"size": "640x360"},
    LLMProvider.TOGETHER: {"seconds": 4, "size": "864x496"},
}


@pytest.mark.asyncio
async def test_video_generation_providers_async(
    provider: LLMProvider,
    video_provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Every video provider starts a job, reports it by id, and finishes it with a downloadable video."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_VIDEO_GENERATION:
            pytest.skip(f"{provider.value} does not support video generation, skipping")

        model_id = video_provider_model_map[provider]
        job = await llm.avideo_generation(model_id, _PROMPT, **_GENERATION_KWARGS.get(provider, {}))
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

    retrieved = await llm.aretrieve_video(job.id)
    assert retrieved.id == job.id

    finished = await llm.apoll_video(job.id, poll_interval=10, timeout=900)
    assert finished.status == "completed", finished.error
    assert finished.outputs

    video = await llm.adownload_video(finished.outputs[0])
    assert len(video) > 1000
