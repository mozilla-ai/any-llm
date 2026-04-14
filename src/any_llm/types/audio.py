"""Audio transcription and speech types for any_llm."""

from typing import IO, Any, Literal

from openai.types.audio import Transcription as OpenAITranscription
from openai.types.audio import TranscriptionVerbose as OpenAITranscriptionVerbose
from pydantic import BaseModel, ConfigDict


class AudioTranscriptionParams(BaseModel):
    """Parameters for audio transcription requests."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model_id: str
    file: bytes | IO[bytes]
    language: str | None = None
    prompt: str | None = None
    response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] | None = None
    temperature: float | None = None
    timestamp_granularities: list[Literal["word", "segment"]] | None = None

    def to_api_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for the provider API call, excluding None values and internal fields."""
        return {k: v for k, v in self.model_dump(exclude={"model_id", "file"}).items() if v is not None}


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
