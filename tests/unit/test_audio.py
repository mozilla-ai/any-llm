import io
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from any_llm import AnyLLM
from any_llm.api import aspeech, atranscription, speech, transcription
from any_llm.constants import LLMProvider
from any_llm.exceptions import InvalidRequestError
from any_llm.types.audio import (
    AudioSpeechParams,
    AudioTranscriptionParams,
    ResolvedAudioFile,
    Transcription,
    sniff_audio_format,
)


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
    expected_supported = {LLMProvider.OPENAI, LLMProvider.AZUREOPENAI, LLMProvider.OTARI}

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


M4A_HEADER = b"\x00\x00\x00\x1cftypM4A \x00\x00\x00\x00M4A mp42isom" + b"\x00" * 16
MP4_HEADER = b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41" + b"\x00" * 16
WEBM_HEADER = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01" + b"\x00" * 16
OGG_HEADER = b"OggS\x00\x02" + b"\x00" * 20
FLAC_HEADER = b"fLaC\x00\x00\x00\x22" + b"\x00" * 16
WAV_HEADER = b"RIFF\x9e\x07\x02\x00WAVEfmt " + b"\x00" * 16
MP3_ID3_HEADER = b"ID3\x04\x00\x00\x00\x00\x00\x23" + b"\x00" * 16
MP3_FRAME_HEADER = b"\xff\xfb\x90\x64" + b"\x00" * 16


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (M4A_HEADER, ("m4a", "audio/mp4")),
        (MP4_HEADER, ("mp4", "audio/mp4")),
        (WEBM_HEADER, ("webm", "audio/webm")),
        (OGG_HEADER, ("ogg", "audio/ogg")),
        (FLAC_HEADER, ("flac", "audio/flac")),
        (WAV_HEADER, ("wav", "audio/wav")),
        (MP3_ID3_HEADER, ("mp3", "audio/mpeg")),
        (MP3_FRAME_HEADER, ("mp3", "audio/mpeg")),
        (b"RIFF\x9e\x07\x02\x00AVI LIST", None),
        (b"FORM\x00\x02\x17PAIFF", None),
        (b"\xff\x1f\x00\x00", None),
        (b"\xff", None),
        (b"", None),
    ],
)
def test_sniff_audio_format(content: bytes, expected: tuple[str, str] | None) -> None:
    assert sniff_audio_format(content) == expected


def test_transcription_params_accepts_bytes() -> None:
    params = AudioTranscriptionParams(model_id="whisper-1", file=b"data")
    assert params.file == b"data"


def test_transcription_params_accepts_bytesio() -> None:
    stream = io.BytesIO(b"data")
    params = AudioTranscriptionParams(model_id="whisper-1", file=stream)
    assert params.file is stream


def test_transcription_params_accepts_open_file_handle(tmp_path: Path) -> None:
    clip = tmp_path / "clip.m4a"
    clip.write_bytes(M4A_HEADER)
    with clip.open("rb") as handle:
        params = AudioTranscriptionParams(model_id="whisper-1", file=handle)
        assert params.file is handle


def test_transcription_params_accepts_named_temporary_file() -> None:
    # tempfile wrappers are neither typing.IO nor io.IOBase instances, only readable.
    with tempfile.NamedTemporaryFile(suffix=".webm") as handle:
        params = AudioTranscriptionParams(model_id="whisper-1", file=handle)
        assert params.file is handle


def test_transcription_params_accepts_path(tmp_path: Path) -> None:
    clip = tmp_path / "clip.m4a"
    for file in (clip, str(clip)):
        params = AudioTranscriptionParams(model_id="whisper-1", file=file)
        assert params.file == file


@pytest.mark.parametrize("file", [42, None, ["bytes"], {"file": b"data"}])
def test_transcription_params_rejects_unreadable_file(file: Any) -> None:
    with pytest.raises(ValidationError, match="file must be bytes, a path, or a readable binary file object"):
        AudioTranscriptionParams(model_id="whisper-1", file=file)


def test_transcription_params_to_api_kwargs_excludes_file_naming() -> None:
    params = AudioTranscriptionParams(
        model_id="whisper-1", file=b"data", filename="clip.m4a", mime_type="audio/mp4", language="en"
    )
    assert params.to_api_kwargs() == {"language": "en"}


def test_resolve_file_named_bytesio_keeps_name_and_derives_content_type() -> None:
    stream = io.BytesIO(M4A_HEADER)
    stream.name = "audio_message.m4a"
    resolved = AudioTranscriptionParams(model_id="whisper-1", file=stream).resolve_file()
    assert resolved == ResolvedAudioFile(filename="audio_message.m4a", content=M4A_HEADER, content_type="audio/mp4")
    assert resolved.multipart == ("audio_message.m4a", M4A_HEADER, "audio/mp4")


def test_resolve_file_strips_directories_from_the_object_name(tmp_path: Path) -> None:
    clip = tmp_path / "nested" / "voice.webm"
    clip.parent.mkdir()
    clip.write_bytes(WEBM_HEADER)
    with clip.open("rb") as handle:
        resolved = AudioTranscriptionParams(model_id="whisper-1", file=handle).resolve_file()
    assert resolved.filename == "voice.webm"
    assert resolved.content == WEBM_HEADER
    assert resolved.content_type == "audio/webm"


def test_resolve_file_reads_path_and_uses_its_name(tmp_path: Path) -> None:
    clip = tmp_path / "clip.ogg"
    clip.write_bytes(OGG_HEADER)
    resolved = AudioTranscriptionParams(model_id="whisper-1", file=clip).resolve_file()
    assert resolved == ResolvedAudioFile(filename="clip.ogg", content=OGG_HEADER, content_type="audio/ogg")
    resolved = AudioTranscriptionParams(model_id="whisper-1", file=str(clip)).resolve_file()
    assert resolved.filename == "clip.ogg"


def test_resolve_file_missing_path_raises_invalid_request(tmp_path: Path) -> None:
    missing = tmp_path / "missing.m4a"
    with pytest.raises(InvalidRequestError, match="Cannot read audio path"):
        AudioTranscriptionParams(model_id="whisper-1", file=missing).resolve_file()


def test_resolve_file_text_mode_handle_raises_invalid_request() -> None:
    with pytest.raises(InvalidRequestError, match="binary mode"):
        AudioTranscriptionParams(model_id="whisper-1", file=io.StringIO("not bytes")).resolve_file()  # type: ignore[arg-type]


def test_resolve_file_unnamed_bytes_get_a_name_from_their_container() -> None:
    resolved = AudioTranscriptionParams(model_id="whisper-1", file=WEBM_HEADER).resolve_file()
    assert resolved == ResolvedAudioFile(filename="audio.webm", content=WEBM_HEADER, content_type="audio/webm")


def test_resolve_file_unnamed_bytesio_gets_a_name_from_its_container() -> None:
    resolved = AudioTranscriptionParams(model_id="whisper-1", file=io.BytesIO(MP3_FRAME_HEADER)).resolve_file()
    assert resolved.filename == "audio.mp3"
    assert resolved.content_type == "audio/mpeg"


def test_resolve_file_handle_with_descriptor_name_falls_back_to_sniffing() -> None:
    # Pipes and sockets expose an int descriptor as name, which is no use as a filename.
    handle = Mock()
    handle.read.return_value = FLAC_HEADER
    handle.name = 7
    resolved = AudioTranscriptionParams(model_id="whisper-1", file=handle).resolve_file()
    assert resolved.filename == "audio.flac"
    assert resolved.content_type == "audio/flac"


def test_resolve_file_unrecognized_bytes_stay_nameless_and_untyped() -> None:
    resolved = AudioTranscriptionParams(model_id="whisper-1", file=b"not audio at all").resolve_file()
    assert resolved == ResolvedAudioFile(filename="audio", content=b"not audio at all", content_type=None)


def test_resolve_file_explicit_filename_wins_over_object_name() -> None:
    stream = io.BytesIO(M4A_HEADER)
    stream.name = "ignored.m4a"
    resolved = AudioTranscriptionParams(model_id="whisper-1", file=stream, filename="clip.mp4").resolve_file()
    assert resolved.filename == "clip.mp4"
    assert resolved.content_type == "audio/mp4"


def test_resolve_file_explicit_filename_names_bytes() -> None:
    resolved = AudioTranscriptionParams(model_id="whisper-1", file=b"opaque", filename="clip.mp3").resolve_file()
    assert resolved == ResolvedAudioFile(filename="clip.mp3", content=b"opaque", content_type="audio/mpeg")


def test_resolve_file_explicit_filename_without_extension_gets_one_from_the_bytes() -> None:
    resolved = AudioTranscriptionParams(model_id="whisper-1", file=WAV_HEADER, filename="voice").resolve_file()
    assert resolved == ResolvedAudioFile(filename="voice.wav", content=WAV_HEADER, content_type="audio/wav")


def test_resolve_file_explicit_mime_type_wins_and_names_the_extension() -> None:
    resolved = AudioTranscriptionParams(
        model_id="whisper-1", file=b"opaque", mime_type="audio/webm; codecs=opus"
    ).resolve_file()
    assert resolved == ResolvedAudioFile(
        filename="audio.webm", content=b"opaque", content_type="audio/webm; codecs=opus"
    )


def test_resolve_file_explicit_mime_type_overrides_the_extension_guess() -> None:
    resolved = AudioTranscriptionParams(
        model_id="whisper-1", file=b"opaque", filename="clip.m4a", mime_type="audio/x-m4a"
    ).resolve_file()
    assert resolved == ResolvedAudioFile(filename="clip.m4a", content=b"opaque", content_type="audio/x-m4a")


def test_resolve_file_unknown_mime_type_leaves_the_name_bare() -> None:
    resolved = AudioTranscriptionParams(model_id="whisper-1", file=b"opaque", mime_type="audio/x-custom").resolve_file()
    assert resolved == ResolvedAudioFile(filename="audio", content=b"opaque", content_type="audio/x-custom")


def test_resolve_file_unknown_extension_gets_the_one_from_the_bytes_appended() -> None:
    # OpenAI decides the format by the extension, so "clip.bin" would be refused even typed audio/ogg.
    resolved = AudioTranscriptionParams(model_id="whisper-1", file=OGG_HEADER, filename="clip.bin").resolve_file()
    assert resolved == ResolvedAudioFile(filename="clip.bin.ogg", content=OGG_HEADER, content_type="audio/ogg")


def test_resolve_file_unknown_extension_gets_the_one_from_the_mime_type_appended() -> None:
    resolved = AudioTranscriptionParams(
        model_id="whisper-1", file=b"opaque", filename="clip.bin", mime_type="audio/webm"
    ).resolve_file()
    assert resolved == ResolvedAudioFile(filename="clip.bin.webm", content=b"opaque", content_type="audio/webm")


def test_resolve_file_known_extension_is_kept_even_when_the_bytes_say_otherwise() -> None:
    # The caller's name is authoritative once it carries a supported audio extension.
    resolved = AudioTranscriptionParams(model_id="whisper-1", file=WEBM_HEADER, filename="clip.mp3").resolve_file()
    assert resolved == ResolvedAudioFile(filename="clip.mp3", content=WEBM_HEADER, content_type="audio/mpeg")


def test_resolve_file_unknown_extension_without_recognizable_bytes_has_no_content_type() -> None:
    resolved = AudioTranscriptionParams(model_id="whisper-1", file=b"opaque", filename="clip.aiff").resolve_file()
    assert resolved == ResolvedAudioFile(filename="clip.aiff", content=b"opaque", content_type=None)


@pytest.mark.asyncio
async def test_atranscription_passes_filename_and_mime_type() -> None:
    mock_provider = Mock()
    mock_provider._atranscription = AsyncMock(return_value=_make_mock_transcription())
    stream = io.BytesIO(b"audio-data")

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider
        await atranscription("openai:whisper-1", file=stream, filename="clip.m4a", mime_type="audio/mp4")

    params = mock_provider._atranscription.call_args[0][0]
    assert isinstance(params, AudioTranscriptionParams)
    assert params.file is stream
    assert params.filename == "clip.m4a"
    assert params.mime_type == "audio/mp4"


def test_sync_transcription_passes_filename_and_mime_type() -> None:
    mock_provider = Mock()
    mock_provider._transcription = Mock(return_value=_make_mock_transcription())

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider
        transcription("openai:whisper-1", file=b"audio-data", filename="clip.m4a", mime_type="audio/mp4")

    kwargs = mock_provider._transcription.call_args.kwargs
    assert kwargs["filename"] == "clip.m4a"
    assert kwargs["mime_type"] == "audio/mp4"


@pytest.mark.asyncio
async def test_anyllm_atranscription_accepts_named_bytesio_and_naming_kwargs() -> None:
    mock_provider = Mock(spec=AnyLLM)
    mock_provider.SUPPORTS_AUDIO_TRANSCRIPTION = True
    mock_provider._atranscription = AsyncMock(return_value=_make_mock_transcription())
    stream = io.BytesIO(b"audio-data")
    stream.name = "audio_message.m4a"

    await AnyLLM.atranscription(mock_provider, "whisper-1", stream, filename="clip.m4a", mime_type="audio/mp4")

    params = mock_provider._atranscription.call_args[0][0]
    assert params.file is stream
    assert params.filename == "clip.m4a"
    assert params.mime_type == "audio/mp4"


def test_resolve_file_unknown_mime_type_still_takes_the_extension_from_the_bytes() -> None:
    resolved = AudioTranscriptionParams(
        model_id="whisper-1", file=WEBM_HEADER, mime_type="audio/x-custom"
    ).resolve_file()
    assert resolved == ResolvedAudioFile(filename="audio.webm", content=WEBM_HEADER, content_type="audio/x-custom")
