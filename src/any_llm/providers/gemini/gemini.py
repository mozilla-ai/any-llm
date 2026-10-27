import io
import os
import re
import wave
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from google import genai
from google.genai import types
from typing_extensions import override

from any_llm.exceptions import MissingApiKeyError, ProviderError, UnsupportedParameterError
from any_llm.types.audio import AudioSpeechParams
from any_llm.types.files import FileOperation
from any_llm.types.responses import Response, ResponsesParams, ResponseStreamEvent
from any_llm.utils.aio import aclose_quietly

from .base import GoogleProvider
from .files import GeminiFileMethods

# A Gemini TTS model answers with one inline blob of headerless 16-bit mono PCM whose MIME type
# carries the rate, such as ``audio/L16;codec=pcm;rate=24000``.
_PCM_RATE_PATTERN = re.compile(r"rate=(\d+)")
_PCM_DEFAULT_RATE = 24000
_PCM_SAMPLE_WIDTH = 2
_PCM_CHANNELS = 1
_SPEECH_RESPONSE_FORMATS = frozenset({"pcm", "wav"})


def _pcm_to_wav(pcm: bytes, mime_type: str | None) -> bytes:
    """Wrap raw 16-bit mono PCM in a RIFF/WAVE container at the rate the MIME type reports."""
    match = _PCM_RATE_PATTERN.search(mime_type or "")
    rate = int(match.group(1)) if match else _PCM_DEFAULT_RATE
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(_PCM_CHANNELS)
        handle.setsampwidth(_PCM_SAMPLE_WIDTH)
        handle.setframerate(rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def _inline_audio(response: types.GenerateContentResponse) -> types.Blob | None:
    """Return the first inline audio blob in the response, or ``None`` when it carries no audio."""
    for candidate in response.candidates or []:
        parts = candidate.content.parts if candidate.content is not None else None
        for part in parts or []:
            if part.inline_data is not None and part.inline_data.data:
                return part.inline_data
    return None


class GeminiProvider(GeminiFileMethods, GoogleProvider):
    """Gemini Provider using the Google GenAI Developer API."""

    PROVIDER_NAME = "gemini"
    PROVIDER_DOCUMENTATION_URL = "https://ai.google.dev/gemini-api/docs"
    ENV_API_KEY_NAME = "GEMINI_API_KEY/GOOGLE_API_KEY"
    ENV_API_BASE_NAME = "GOOGLE_GEMINI_BASE_URL"
    SUPPORTS_RESPONSES = True
    SUPPORTS_AUDIO_SPEECH = True
    SUPPORTED_FILE_OPERATIONS: ClassVar[frozenset[FileOperation]] = frozenset({"upload", "list", "retrieve", "delete"})

    _interactions_api_version: str | None

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise MissingApiKeyError(self.PROVIDER_NAME, self.ENV_API_KEY_NAME)
        return api_key

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        http_options = kwargs.get("http_options")
        if isinstance(http_options, dict):
            configured_api_version = http_options.get("api_version")
        elif isinstance(http_options, types.HttpOptions):
            configured_api_version = http_options.api_version
        else:
            configured_api_version = None
        # Follow the client's configured version, leaving None to the SDK default of
        # v1beta. Interactions is served on both v1 and v1beta, but v1 does not carry
        # preview models, so pinning it here would reject models completion() accepts.
        # https://ai.google.dev/gemini-api/docs/api-versions
        self._interactions_api_version = configured_api_version

        if api_base:
            http_options = kwargs.pop("http_options", None)
            if http_options is None:
                http_options = types.HttpOptions(base_url=api_base)
            elif isinstance(http_options, dict):
                http_options.setdefault("base_url", api_base)
            elif isinstance(http_options, types.HttpOptions) and http_options.base_url is None:
                http_options.base_url = api_base
            kwargs["http_options"] = http_options

        # Ensure timeout is correctly configured if present.
        if (timeout := kwargs.pop("timeout", None)) is not None:
            GoogleProvider._merge_timeout_into_http_options(timeout, kwargs)

        self.client = genai.Client(api_key=api_key, **kwargs)

    @override
    async def _aresponses(
        self, params: ResponsesParams, **kwargs: Any
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        if kwargs.pop("extra_body", None) is not None:
            parameter_name = "extra_body"
            raise UnsupportedParameterError(parameter_name, self.PROVIDER_NAME)
        timeout = kwargs.pop("timeout", None)
        create_kwargs = {name: value for name, value in kwargs.items() if value is not None}
        unsupported_parameters = create_kwargs.keys() - {"extra_headers", "extra_query"}
        if unsupported_parameters:
            parameter_name = min(unsupported_parameters)
            raise UnsupportedParameterError(parameter_name, self.PROVIDER_NAME)

        # Vertex shares this package without requiring the Interactions SDK.
        from .interactions import convert_interaction_to_response, convert_responses_params

        create_kwargs = (
            convert_responses_params(
                params,
                self.PROVIDER_NAME,
                api_version=self._interactions_api_version,
            )
            | create_kwargs
        )
        if timeout is not None:
            create_kwargs["timeout"] = timeout
        if params.stream:
            return self._create_interaction_stream(create_kwargs, model=params.model)
        interaction = await self.client.aio.interactions.create(**create_kwargs)
        return convert_interaction_to_response(interaction, fallback_model=params.model)

    async def _create_interaction_stream(
        self, create_kwargs: dict[str, Any], *, model: str
    ) -> AsyncIterator[ResponseStreamEvent]:
        from .interactions_stream import convert_interaction_stream

        stream = await self.client.aio.interactions.create(**create_kwargs)
        converted_stream = convert_interaction_stream(stream, model=model)
        try:
            async for event in converted_stream:
                yield event
        finally:
            await aclose_quietly(converted_stream)

    @override
    async def _aspeech(self, params: AudioSpeechParams, **kwargs: Any) -> bytes:
        """Generate speech with a Gemini TTS model through ``generateContent``.

        Gemini has no speech endpoint. Its TTS models (``gemini-2.5-flash-preview-tts``,
        ``gemini-2.5-pro-preview-tts``, ``gemini-3.1-flash-tts-preview``) answer a
        ``generateContent`` call whose config asks for the audio modality and names one of the
        prebuilt voices such as ``Kore``; further ``GenerateContentConfig`` fields pass through
        ``kwargs``. The reply is headerless 16-bit mono PCM, returned as-is for ``pcm`` and
        wrapped in a WAV container for ``wav``, the default. Gemini encodes nothing else, so mp3,
        opus, aac and flac raise ``UnsupportedParameterError``. Delivery is steered by the text
        itself (``Say cheerfully: ...``) and there is no rate control, so ``instructions`` and
        ``speed`` raise too. See https://ai.google.dev/gemini-api/docs/speech-generation.
        """
        if params.instructions is not None:
            parameter_name = "instructions"
            raise UnsupportedParameterError(
                parameter_name,
                self.PROVIDER_NAME,
                "Gemini takes style direction as part of the text itself, for example `Say cheerfully: ...`.",
            )
        if params.speed is not None:
            parameter_name = "speed"
            raise UnsupportedParameterError(parameter_name, self.PROVIDER_NAME)
        response_format = params.response_format or "wav"
        if response_format not in _SPEECH_RESPONSE_FORMATS:
            parameter_name = "response_format"
            raise UnsupportedParameterError(
                parameter_name,
                self.PROVIDER_NAME,
                f"Gemini speech returns {', '.join(sorted(_SPEECH_RESPONSE_FORMATS))}, not {response_format!r}.",
            )

        config: dict[str, Any] = {
            "response_modalities": ["AUDIO"],
            "speech_config": types.SpeechConfig(
                voice_config=types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=params.voice))
            ),
        }
        config.update(kwargs)
        response: types.GenerateContentResponse = await self.client.aio.models.generate_content(
            model=params.model_id,
            contents=params.input,
            config=types.GenerateContentConfig(**config),
        )
        audio = _inline_audio(response)
        if audio is None or audio.data is None:
            finish_reason = response.candidates[0].finish_reason if response.candidates else None
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else None
            msg = (
                f"Gemini returned no audio for model {params.model_id!r} "
                f"(finish_reason={finish_reason}, block_reason={block_reason})."
            )
            raise ProviderError(msg, provider_name=self.PROVIDER_NAME)
        if response_format == "pcm":
            return audio.data
        return _pcm_to_wav(audio.data, audio.mime_type)
