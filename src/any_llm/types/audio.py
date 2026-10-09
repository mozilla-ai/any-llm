"""Audio transcription and speech types for any_llm."""

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal

from openai.types.audio import Transcription as OpenAITranscription
from openai.types.audio import TranscriptionVerbose as OpenAITranscriptionVerbose
from pydantic import BaseModel, ConfigDict, PlainValidator

from any_llm.exceptions import InvalidRequestError
from any_llm.types.files import FileInput

# Fallback name for audio that arrives without one. Multipart file parts need a filename, and
# providers that cannot tell the format from a bare name reject the upload, so the name gets
# an extension whenever the format is known.
DEFAULT_AUDIO_FILENAME = "audio"

# Content type per extension for the formats the transcription endpoints document. Kept here
# instead of taken from mimetypes because the host database varies (macOS maps .m4a to
# audio/mp4a-latm; minimal Linux images have no entry for .webm).
_AUDIO_MIME_TYPES: dict[str, str] = {
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "mp3": "audio/mpeg",
    "mp4": "audio/mp4",
    "mpeg": "audio/mpeg",
    "mpga": "audio/mpeg",
    "oga": "audio/ogg",
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
    "wav": "audio/wav",
    "webm": "audio/webm",
}

# Content type (including common aliases) to the extension used when a name has none.
_AUDIO_EXTENSIONS: dict[str, str] = {
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/ogg": "ogg",
    "audio/opus": "ogg",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/webm": "webm",
    "video/webm": "webm",
    "video/mp4": "mp4",
}

# Container signatures of the formats the transcription endpoints accept, so bytes that arrive
# without a name can still be uploaded as a recognizable audio file.
_AUDIO_SIGNATURES: tuple[tuple[int, bytes, str], ...] = (
    (0, b"fLaC", "flac"),
    (0, b"ID3", "mp3"),
    (0, b"OggS", "ogg"),
    (0, b"\x1a\x45\xdf\xa3", "webm"),
    (4, b"ftypM4A", "m4a"),
    (4, b"ftyp", "mp4"),
)


def sniff_audio_format(content: bytes) -> tuple[str, str] | None:
    """Return ``(extension, content_type)`` for audio bytes whose container can be recognized.

    Covers the containers the transcription endpoints accept: flac, mp3, mp4/m4a, ogg, wav and
    webm. Returns ``None`` for anything else, including raw PCM.
    """
    extension: str | None = None
    for offset, magic, candidate in _AUDIO_SIGNATURES:
        if content[offset : offset + len(magic)] == magic:
            extension = candidate
            break
    if extension is None and content[:4] == b"RIFF" and content[8:12] == b"WAVE":
        extension = "wav"
    # An MP3 without an ID3 tag starts straight at a frame header: 11 sync bits set.
    if extension is None and len(content) >= 2 and content[0] == 0xFF and content[1] & 0xE0 == 0xE0:
        extension = "mp3"
    if extension is None:
        return None
    return extension, _AUDIO_MIME_TYPES[extension]


def _validate_audio_input(value: Any) -> Any:
    """Accept bytes, a path, or any object with a ``read`` method as transcription input."""
    # typing.BinaryIO is not usable with isinstance (io.BytesIO and open() handles both fail it),
    # so accept anything readable instead of the annotation's exact types.
    if isinstance(value, (bytes, str, PathLike)) or callable(getattr(value, "read", None)):
        return value
    msg = "file must be bytes, a path, or a readable binary file object"
    raise ValueError(msg)


AudioInput = Annotated[FileInput, PlainValidator(_validate_audio_input)]


@dataclass(frozen=True)
class ResolvedAudioFile:
    """Audio content together with the multipart file name and content type to upload it under."""

    filename: str
    content: bytes
    content_type: str | None

    @property
    def multipart(self) -> tuple[str, bytes, str | None]:
        """The ``(filename, content, content_type)`` triple the OpenAI SDK takes for a file part."""
        return (self.filename, self.content, self.content_type)


class AudioTranscriptionParams(BaseModel):
    """Parameters for audio transcription requests."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    file: AudioInput
    filename: str | None = None
    mime_type: str | None = None
    language: str | None = None
    prompt: str | None = None
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] | None = None
    temperature: float | None = None
    timestamp_granularities: list[Literal["word", "segment"]] | None = None

    def to_api_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for the provider API call, excluding None values and internal fields."""
        return {
            k: v
            for k, v in self.model_dump(exclude={"model_id", "file", "filename", "mime_type"}).items()
            if v is not None
        }

    def resolve_file(self) -> ResolvedAudioFile:
        """Read ``file`` and work out the name and content type its multipart part should carry.

        The name comes from ``filename``, else the path or the file object's ``name``, else the
        audio container recognized from the bytes. The content type comes from ``mime_type``,
        else the name's extension, else the recognized container. OpenAI decides the format by
        the extension alone (``voice.bin`` with type audio/webm is refused, ``voice.bin.webm`` is
        not), so a name without a known audio extension gets the one for the content type or the
        recognized container appended.
        """
        content, name = self._read_file()
        sniffed = sniff_audio_format(content)
        content_type = self.mime_type
        if name is None:
            name = DEFAULT_AUDIO_FILENAME
        extension = Path(name).suffix[1:].lower()
        if content_type is None and extension:
            content_type = _AUDIO_MIME_TYPES.get(extension)
        if content_type is None and sniffed is not None:
            content_type = sniffed[1]
        if extension not in _AUDIO_MIME_TYPES:
            new_extension = None
            if content_type is not None:
                new_extension = _AUDIO_EXTENSIONS.get(content_type.split(";")[0].strip().lower())
            if new_extension is None and sniffed is not None:
                new_extension = sniffed[0]
            if new_extension is not None:
                name = f"{name}.{new_extension}"
        return ResolvedAudioFile(filename=name, content=content, content_type=content_type)

    def _read_file(self) -> tuple[bytes, str | None]:
        """Return the audio bytes and the file name they came with, if any."""
        if isinstance(self.file, bytes):
            return self.file, self.filename
        if isinstance(self.file, (str, PathLike)):
            path = Path(self.file)
            try:
                return path.read_bytes(), self.filename or path.name
            except (OSError, ValueError) as exc:
                # Without this, exception conversion classifies FileNotFoundError by its type name as
                # ModelNotFoundError. ValueError is what an embedded NUL byte in the path raises.
                msg = f"Cannot read audio path {str(path)!r}: {getattr(exc, 'strerror', None) or exc}"
                raise InvalidRequestError(msg, original_exception=exc) from exc
        content = self.file.read()
        if not isinstance(content, bytes):
            msg = f"file must be opened in binary mode, got {type(content).__name__} from read()"
            raise InvalidRequestError(msg)
        name = self.filename
        if name is None:
            # Handles opened from a path carry the path; sockets and pipes carry an int descriptor.
            object_name = getattr(self.file, "name", None)
            if isinstance(object_name, str) and object_name:
                name = Path(object_name).name
        return content, name


class AudioSpeechParams(BaseModel):
    """Parameters for audio speech (TTS) requests."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    input: str
    voice: str
    instructions: str | None = None
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] | None = None
    speed: float | None = None

    def to_api_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for the provider API call, excluding None values and internal fields."""
        return {k: v for k, v in self.model_dump(exclude={"model_id", "input", "voice"}).items() if v is not None}


# Re-export OpenAI types for convenience
Transcription = OpenAITranscription
TranscriptionVerbose = OpenAITranscriptionVerbose
