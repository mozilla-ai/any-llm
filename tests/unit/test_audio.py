from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm import AnyLLM
from any_llm.api import aspeech, atranscription, speech, transcription
from any_llm.constants import LLMProvider
from any_llm.types.audio import AudioSpeechParams, AudioTranscriptionParams, Transcription


def _make_mock_transcription() -> Transcription:
    return Transcription(text="Hello, world!")


FAKE_AUDIO_BYTES = b"fake-audio-content"


@pytest.mark.asyncio
async def test_atranscription_with_api_config() -> None:
    mock_provider = Mock()
    mock_response = _make_mock_transcription()
    mock_provider._atranscription = AsyncMock(return_value=mock_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        result = await atranscription(
            "openai:whisper-1",
            file=b"audio-data",
            api_key="test_key",
            api_base="https://test.example.com",
        )

        call_args = mock_create.call_args
        assert call_args[0][0] == LLMProvider.OPENAI
        assert call_args[1]["api_key"] == "test_key"
        assert call_args[1]["api_base"] == "https://test.example.com"

        mock_provider._atranscription.assert_called_once()
        params = mock_provider._atranscription.call_args[0][0]
        assert isinstance(params, AudioTranscriptionParams)
        assert params.model_id == "whisper-1"
        assert params.file == b"audio-data"
        assert result == mock_response


@pytest.mark.asyncio
async def test_atranscription_with_explicit_provider() -> None:
    mock_provider = Mock()
    mock_response = _make_mock_transcription()
    mock_provider._atranscription = AsyncMock(return_value=mock_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        result = await atranscription(
            "whisper-1",
            file=b"audio-data",
            provider="openai",
            language="en",
        )

        call_args = mock_create.call_args
        assert call_args[0][0] == LLMProvider.OPENAI

        params = mock_provider._atranscription.call_args[0][0]
        assert params.model_id == "whisper-1"
        assert params.language == "en"
        assert result == mock_response


@pytest.mark.asyncio
async def test_atranscription_passes_all_params() -> None:
    mock_provider = Mock()
    mock_response = _make_mock_transcription()
    mock_provider._atranscription = AsyncMock(return_value=mock_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        await atranscription(
            "openai:whisper-1",
            file=b"audio-data",
            language="en",
            prompt="Previous segment context",
            response_format="verbose_json",
            temperature=0.2,
            timestamp_granularities=["word", "segment"],
        )

        params = mock_provider._atranscription.call_args[0][0]
        assert params.language == "en"
        assert params.prompt == "Previous segment context"
        assert params.response_format == "verbose_json"
        assert params.temperature == 0.2
        assert params.timestamp_granularities == ["word", "segment"]


def test_sync_transcription_dispatches() -> None:
    mock_provider = Mock()
    mock_response = _make_mock_transcription()
    mock_provider._transcription = Mock(return_value=mock_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        result = transcription(
            "openai:whisper-1",
            file=b"audio-data",
            api_key="test_key",
        )

        mock_provider._transcription.assert_called_once()
        assert result == mock_response


@pytest.mark.asyncio
async def test_aspeech_with_api_config() -> None:
    mock_provider = Mock()
    mock_provider._aspeech = AsyncMock(return_value=FAKE_AUDIO_BYTES)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        result = await aspeech(
            "openai:tts-1",
            input="Hello, world!",
            voice="alloy",
            api_key="test_key",
            api_base="https://test.example.com",
        )

        call_args = mock_create.call_args
        assert call_args[0][0] == LLMProvider.OPENAI
        assert call_args[1]["api_key"] == "test_key"

        mock_provider._aspeech.assert_called_once()
        params = mock_provider._aspeech.call_args[0][0]
        assert isinstance(params, AudioSpeechParams)
        assert params.model_id == "tts-1"
        assert params.input == "Hello, world!"
        assert params.voice == "alloy"
        assert result == FAKE_AUDIO_BYTES


@pytest.mark.asyncio
async def test_aspeech_with_explicit_provider() -> None:
    mock_provider = Mock()
    mock_provider._aspeech = AsyncMock(return_value=FAKE_AUDIO_BYTES)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        result = await aspeech(
            "tts-1",
            input="Hi",
            voice="echo",
            provider="openai",
            response_format="opus",
            speed=1.5,
        )

        call_args = mock_create.call_args
        assert call_args[0][0] == LLMProvider.OPENAI

        params = mock_provider._aspeech.call_args[0][0]
        assert params.model_id == "tts-1"
        assert params.input == "Hi"
        assert params.voice == "echo"
        assert params.response_format == "opus"
        assert params.speed == 1.5
        assert result == FAKE_AUDIO_BYTES


@pytest.mark.asyncio
async def test_aspeech_passes_all_params() -> None:
    mock_provider = Mock()
    mock_provider._aspeech = AsyncMock(return_value=FAKE_AUDIO_BYTES)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        await aspeech(
            "openai:tts-1",
            input="Generate this speech",
            voice="shimmer",
            instructions="Speak slowly and clearly",
            response_format="flac",
            speed=0.75,
        )

        params = mock_provider._aspeech.call_args[0][0]
        assert params.input == "Generate this speech"
        assert params.voice == "shimmer"
        assert params.instructions == "Speak slowly and clearly"
        assert params.response_format == "flac"
        assert params.speed == 0.75


def test_sync_speech_dispatches() -> None:
    mock_provider = Mock()
    mock_provider._speech = Mock(return_value=FAKE_AUDIO_BYTES)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        result = speech(
            "openai:tts-1",
            input="Hello",
            voice="alloy",
            api_key="test_key",
        )

        mock_provider._speech.assert_called_once()
        assert result == FAKE_AUDIO_BYTES


def test_sync_transcription_with_explicit_provider() -> None:
    mock_provider = Mock()
    mock_response = _make_mock_transcription()
    mock_provider._transcription = Mock(return_value=mock_response)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        result = transcription(
            "whisper-1",
            file=b"audio-data",
            provider="openai",
            language="en",
        )

        call_args = mock_create.call_args
        assert call_args[0][0] == LLMProvider.OPENAI
        mock_provider._transcription.assert_called_once()
        assert result == mock_response


def test_sync_speech_with_explicit_provider() -> None:
    mock_provider = Mock()
    mock_provider._speech = Mock(return_value=FAKE_AUDIO_BYTES)

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider

        result = speech(
            "tts-1",
            input="Hello",
            voice="alloy",
            provider="openai",
            instructions="Be clear",
        )

        call_args = mock_create.call_args
        assert call_args[0][0] == LLMProvider.OPENAI
        mock_provider._speech.assert_called_once()
        assert result == FAKE_AUDIO_BYTES


@pytest.mark.asyncio
async def test_anyllm_atranscription_constructs_params() -> None:
    mock_provider = Mock(spec=AnyLLM)
    mock_provider.SUPPORTS_AUDIO_TRANSCRIPTION = True
    mock_response = _make_mock_transcription()
    mock_provider._atranscription = AsyncMock(return_value=mock_response)
    mock_provider.PROVIDER_NAME = "openai"

    result = await AnyLLM.atranscription(mock_provider, model="whisper-1", file=b"data", language="en")

    mock_provider._atranscription.assert_called_once()
    params = mock_provider._atranscription.call_args[0][0]
    assert isinstance(params, AudioTranscriptionParams)
    assert params.model_id == "whisper-1"
    assert params.file == b"data"
    assert params.language == "en"
    assert result == mock_response


@pytest.mark.asyncio
async def test_anyllm_aspeech_constructs_params() -> None:
    mock_provider = Mock(spec=AnyLLM)
    mock_provider.SUPPORTS_AUDIO_SPEECH = True
    mock_provider._aspeech = AsyncMock(return_value=FAKE_AUDIO_BYTES)
    mock_provider.PROVIDER_NAME = "openai"

    result = await AnyLLM.aspeech(mock_provider, model="tts-1", input="hello", voice="alloy", speed=1.5)

    mock_provider._aspeech.assert_called_once()
    params = mock_provider._aspeech.call_args[0][0]
    assert isinstance(params, AudioSpeechParams)
    assert params.model_id == "tts-1"
    assert params.input == "hello"
    assert params.voice == "alloy"
    assert params.speed == 1.5
    assert result == FAKE_AUDIO_BYTES


@pytest.mark.asyncio
async def test_atranscription_unsupported_provider_raises() -> None:
    params = AudioTranscriptionParams(model_id="some-model", file=b"data")
    base = Mock(spec=AnyLLM)
    base.SUPPORTS_AUDIO_TRANSCRIPTION = False
    with pytest.raises(NotImplementedError, match="doesn't support audio transcription"):
        await AnyLLM._atranscription(base, params)


@pytest.mark.asyncio
async def test_aspeech_unsupported_provider_raises() -> None:
    params = AudioSpeechParams(model_id="some-model", input="hi", voice="alloy")
    base = Mock(spec=AnyLLM)
    base.SUPPORTS_AUDIO_SPEECH = False
    with pytest.raises(NotImplementedError, match="doesn't support audio speech"):
        await AnyLLM._aspeech(base, params)


@pytest.mark.asyncio
async def test_atranscription_supported_but_not_implemented_raises() -> None:
    params = AudioTranscriptionParams(model_id="some-model", file=b"data")
    base = Mock(spec=AnyLLM)
    base.SUPPORTS_AUDIO_TRANSCRIPTION = True
    with pytest.raises(NotImplementedError, match="Subclasses must implement _atranscription"):
        await AnyLLM._atranscription(base, params)


@pytest.mark.asyncio
async def test_aspeech_supported_but_not_implemented_raises() -> None:
    params = AudioSpeechParams(model_id="some-model", input="hi", voice="alloy")
    base = Mock(spec=AnyLLM)
    base.SUPPORTS_AUDIO_SPEECH = True
    with pytest.raises(NotImplementedError, match="Subclasses must implement _aspeech"):
        await AnyLLM._aspeech(base, params)


def test_transcription_params_to_api_kwargs_excludes_none() -> None:
    params = AudioTranscriptionParams(model_id="whisper-1", file=b"data")
    kwargs = params.to_api_kwargs()
    assert "model_id" not in kwargs
    assert "file" not in kwargs
    assert kwargs == {}


def test_transcription_params_to_api_kwargs_includes_set_values() -> None:
    params = AudioTranscriptionParams(
        model_id="whisper-1",
        file=b"data",
        language="en",
        prompt="context",
        response_format="verbose_json",
        temperature=0.3,
        timestamp_granularities=["word"],
    )
    kwargs = params.to_api_kwargs()
    assert "model_id" not in kwargs
    assert "file" not in kwargs
    assert kwargs == {
        "language": "en",
        "prompt": "context",
        "response_format": "verbose_json",
        "temperature": 0.3,
        "timestamp_granularities": ["word"],
    }


def test_speech_params_to_api_kwargs_excludes_none() -> None:
    params = AudioSpeechParams(model_id="tts-1", input="hi", voice="alloy")
    kwargs = params.to_api_kwargs()
    assert "model_id" not in kwargs
    assert "input" not in kwargs
    assert "voice" not in kwargs
    assert kwargs == {}


def test_speech_params_to_api_kwargs_includes_set_values() -> None:
    params = AudioSpeechParams(
        model_id="tts-1",
        input="hello",
        voice="echo",
        instructions="Speak fast",
        response_format="opus",
        speed=2.0,
    )
    kwargs = params.to_api_kwargs()
    assert "model_id" not in kwargs
    assert "input" not in kwargs
    assert "voice" not in kwargs
    assert kwargs == {
        "instructions": "Speak fast",
        "response_format": "opus",
        "speed": 2.0,
    }


def test_transcription_params_rejects_extra_fields() -> None:
    with pytest.raises(Exception, match="extra"):
        AudioTranscriptionParams(model_id="whisper-1", file=b"data", bogus="value")  # type: ignore[call-arg]


def test_speech_params_rejects_extra_fields() -> None:
    with pytest.raises(Exception, match="extra"):
        AudioSpeechParams(model_id="tts-1", input="hi", voice="alloy", bogus="value")  # type: ignore[call-arg]


def test_supports_audio_only_on_expected_providers() -> None:
    expected_supported = {LLMProvider.OPENAI, LLMProvider.AZUREOPENAI, LLMProvider.GATEWAY}

    for provider_enum in expected_supported:
        cls = AnyLLM.get_provider_class(provider_enum)
        assert cls.SUPPORTS_AUDIO_TRANSCRIPTION is True, f"{provider_enum.value} should support audio transcription"
        assert cls.SUPPORTS_AUDIO_SPEECH is True, f"{provider_enum.value} should support audio speech"

    for provider_enum in LLMProvider:
        if provider_enum in expected_supported:
            continue
        try:
            cls = AnyLLM.get_provider_class(provider_enum)
        except ImportError:
            continue
        assert cls.SUPPORTS_AUDIO_TRANSCRIPTION is False, (
            f"{provider_enum.value} should not support audio transcription"
        )
        assert cls.SUPPORTS_AUDIO_SPEECH is False, f"{provider_enum.value} should not support audio speech"
