import io
import math
import struct
import wave
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import APIConnectionError, OpenAIError

from any_llm import AnyLLM, LLMProvider
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.audio import Transcription


@pytest.fixture
def transcription_provider_model_map() -> dict[LLMProvider, str]:
    return {
        LLMProvider.OPENAI: "whisper-1",
        LLMProvider.AZUREOPENAI: "whisper",
        LLMProvider.OTARI: "openai:whisper-1",
    }


@pytest.fixture(scope="module")
def tone_wav() -> bytes:
    """One second of a 440 Hz tone as 16 kHz mono 16-bit PCM WAV: real audio without needing a clip on disk."""
    sample_rate = 16000
    frames = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate))) for i in range(sample_rate)
    )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(frames)
    return buffer.getvalue()


def _create_transcription_provider(
    provider: LLMProvider,
    provider_client_config: dict[LLMProvider, dict[str, Any]],
    model_map: dict[LLMProvider, str],
) -> tuple[AnyLLM, str]:
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
    except ImportError:
        pytest.skip(f"{provider.value} optional dependency missing, skipping")
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except OpenAIError as exc:
        pytest.skip(f"{provider.value} client init failed: {exc}")
    except (ValueError, TypeError) as exc:
        pytest.skip(f"{provider.value} requires additional config to instantiate: {exc}")
    except Exception as exc:
        if type(exc).__name__ in {
            "NoRegionError",
            "NoCredentialsError",
            "ProfileNotFound",
            "DefaultCredentialsError",
        }:
            pytest.skip(f"{provider.value} requires additional config to instantiate: {exc}")
        raise

    if not llm.SUPPORTS_AUDIO_TRANSCRIPTION:
        pytest.skip(f"{provider.value} does not support audio transcription, skipping")

    model_id = model_map.get(provider)
    if model_id is None:
        pytest.skip(f"No transcription model mapped for {provider.value}, skipping")
    return llm, model_id


async def _transcribe(llm: AnyLLM, provider: LLMProvider, model_id: str, file: Any, **kwargs: Any) -> Transcription:
    try:
        return await llm.atranscription(model_id, file, **kwargs)
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        pytest.skip(f"{provider.value} connection failed, skipping")
    except Exception as exc:
        if "model" in str(exc).lower() or "deployment" in str(exc).lower():
            pytest.skip(f"{provider.value} transcription model not available: {exc}")
        raise


@pytest.mark.asyncio
async def test_transcription_named_file_object(
    provider: LLMProvider,
    transcription_provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
    tone_wav: bytes,
) -> None:
    """A named in-memory file, the shape web servers hand over, reaches the provider with its name."""
    llm, model_id = _create_transcription_provider(provider, provider_client_config, transcription_provider_model_map)
    stream = io.BytesIO(tone_wav)
    stream.name = "voice_message.wav"

    result = await _transcribe(llm, provider, model_id, stream)

    assert isinstance(result, Transcription)
    assert isinstance(result.text, str)


@pytest.mark.asyncio
async def test_transcription_bare_bytes(
    provider: LLMProvider,
    transcription_provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
    tone_wav: bytes,
) -> None:
    """Bytes without a name are uploaded as the audio format their header shows."""
    llm, model_id = _create_transcription_provider(provider, provider_client_config, transcription_provider_model_map)

    result = await _transcribe(llm, provider, model_id, tone_wav)

    assert isinstance(result, Transcription)
    assert isinstance(result.text, str)


@pytest.mark.asyncio
async def test_transcription_path(
    provider: LLMProvider,
    transcription_provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
    tone_wav: bytes,
    tmp_path: Path,
) -> None:
    llm, model_id = _create_transcription_provider(provider, provider_client_config, transcription_provider_model_map)
    clip = tmp_path / "voice_message.wav"
    clip.write_bytes(tone_wav)

    result = await _transcribe(llm, provider, model_id, clip, language="en")

    assert isinstance(result, Transcription)
    assert isinstance(result.text, str)
