import io
import wave
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.azureopenai.azureopenai import AzureopenaiProvider
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [pytest.mark.asyncio, pytest.mark.parametrize("provider", [LLMProvider.AZUREOPENAI])]


@pytest_asyncio.fixture
async def azure_v1(
    provider: LLMProvider,
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> AsyncIterator[AzureopenaiProvider]:
    try:
        llm = AzureopenaiProvider(**provider_client_config[provider])
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip("Azure v1 credentials missing: set AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN")
    try:
        yield llm
    finally:
        await llm.client.close()


async def test_azure_v1_core(
    azure_v1: AzureopenaiProvider,
    provider_model_map: dict[LLMProvider, str],
) -> None:
    model = provider_model_map[LLMProvider.AZUREOPENAI]
    assert azure_v1.client.base_url.path.endswith("/openai/v1/")
    result = await azure_v1.acompletion(model=model, messages=[{"role": "user", "content": "Say hello."}])
    assert result.choices[0].message.content
    response = await azure_v1.aresponses(model=model, input_data="Say hello.")
    assert response.output
    assert await azure_v1.alist_models()


@pytest.mark.parametrize("operation", ["image", "transcription", "speech"])
async def test_azure_v1_media(
    azure_v1: AzureopenaiProvider,
    operation: str,
    azure_media_model_map: dict[str, str],
) -> None:
    deployment = azure_media_model_map[operation]
    if operation == "image":
        image = await azure_v1.aimage_generation(
            model=deployment, prompt="A small blue circle on a white background.", n=1, quality="low", size="1024x1024"
        )
        assert image.data
    elif operation == "transcription":
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\x00\x00" * 16000)
        buffer.seek(0)
        buffer.name = "silence.wav"
        transcription = await azure_v1.atranscription(model=deployment, file=buffer)
        assert isinstance(transcription.text, str)
    else:
        speech = await azure_v1.aspeech(model=deployment, input="Hello from Azure.", voice="alloy")
        assert speech
