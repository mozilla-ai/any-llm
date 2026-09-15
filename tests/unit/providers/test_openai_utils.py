import warnings

import pytest
from openai.lib._parsing._completions import parse_chat_completion
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from pydantic import BaseModel

from any_llm.exceptions import ProviderError
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.openai.utils import _convert_chat_completion, _convert_parsed_chat_completion
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk


class Answer(BaseModel):
    answer: str


def test_convert_parsed_chat_completion_does_not_emit_serializer_warning() -> None:
    """The SDK's parsed response path must not warn while preserving parsed data."""
    raw = OpenAIChatCompletion.model_validate(
        {
            "id": "chatcmpl-offline",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"answer":"Earth"}'},
                }
            ],
        }
    )
    parsed = parse_chat_completion(response_format=Answer, input_tools=[], chat_completion=raw)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        converted = _convert_parsed_chat_completion(parsed)

    assert not any("Pydantic serializer warnings" in str(item.message) for item in caught)
    assert converted.choices[0].message.parsed == Answer(answer="Earth")


def test_convert_chat_completion_with_empty_response() -> None:
    # Simulating the malformed response described in the PR
    # ChatCompletion(id=None, choices=None, created=None, model=None, object='chat.completion', ...)
    openai_response = OpenAIChatCompletion.model_construct(
        id=None,
        choices=None,
        created=None,
        model=None,
        object="chat.completion",
    )

    with pytest.raises(ProviderError) as exc_info:
        _convert_chat_completion(openai_response)

    assert "Provider returned an empty response" in str(exc_info.value)


def test_convert_chat_completion_with_partial_none_response() -> None:
    # If not all THREE (id, choices, model) are None, it should NOT raise ProviderError early.
    # It might fail later if other required fields like 'created' are missing or invalid,
    # but the specific guard being tested here should not trigger.
    openai_response = OpenAIChatCompletion.model_construct(
        id="test-id",
        choices=None,
        created=1234567890,
        model=None,
        object="chat.completion",
    )

    # In this case, it will fail later during ChatCompletion.model_validate(normalized)
    # because 'choices' is None, or during _normalize_openai_dict_response if it expect choices to be a list.

    # Actually _normalize_openai_dict_response handles choices=None:
    # choices = response_dict.get("choices")
    # if isinstance(choices, list): ...

    # But ChatCompletion.model_validate(normalized) will fail because choices is required.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _convert_chat_completion(openai_response)


def test_chat_completion_accepts_nonstandard_service_tier() -> None:
    """Providers like OpenRouter may return service_tier values outside the OpenAI literal set."""
    completion = ChatCompletion.model_validate(
        {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello"},
                }
            ],
            "service_tier": "standard",
        }
    )
    assert completion.service_tier == "standard"


def test_chat_completion_chunk_accepts_nonstandard_service_tier() -> None:
    """Providers like OpenRouter may return service_tier values outside the OpenAI literal set."""
    chunk = ChatCompletionChunk.model_validate(
        {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hi"},
                    "finish_reason": None,
                }
            ],
            "service_tier": "standard",
        }
    )
    assert chunk.service_tier == "standard"


def test_convert_completion_response_with_nonstandard_service_tier() -> None:
    """The full conversion pipeline should handle non-standard service_tier values."""
    openai_response = OpenAIChatCompletion.model_construct(
        id="test-id",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello"},
            }
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
        service_tier="standard",
    )
    result = _convert_chat_completion(openai_response)
    assert result.service_tier == "standard"


def test_convert_chunk_response_with_nonstandard_service_tier() -> None:
    """The chunk conversion pipeline should handle non-standard service_tier values."""
    openai_chunk = OpenAIChatCompletionChunk.model_construct(
        id="test-id",
        choices=[
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "Hi"},
                "finish_reason": None,
            }
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
        service_tier="standard",
    )
    result = BaseOpenAIProvider._convert_completion_chunk_response(openai_chunk)
    assert result.service_tier == "standard"


@pytest.mark.parametrize(
    "audio_piece",
    [
        {"id": "audio_6a98e62e", "transcript": "Sure"},
        {"data": "AAAB//8CAP=="},
        {"expires_at": 1788408898},
    ],
)
def test_convert_chunk_response_keeps_partial_audio_delta(audio_piece: dict[str, object]) -> None:
    """OpenAI streams delta audio fields in separate chunks."""
    openai_chunk = OpenAIChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-audio",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "gpt-audio-mini",
            "choices": [{"index": 0, "delta": {"role": "assistant", "audio": audio_piece}, "finish_reason": None}],
        }
    )

    result = BaseOpenAIProvider._convert_completion_chunk_response(openai_chunk)

    audio = result.choices[0].delta.audio
    assert audio is not None
    assert audio.model_dump(exclude_none=True) == audio_piece
