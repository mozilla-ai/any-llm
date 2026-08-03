import pytest
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from pydantic import ValidationError

from any_llm.exceptions import ProviderError
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.openai.utils import _convert_chat_completion, _normalize_openai_dict_response
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk


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


def test_convert_chunk_response_maps_model_context_window_exceeded_to_length() -> None:
    """z.ai's GLM models end a truncated stream with a finish reason outside the OpenAI literal set."""
    openai_chunk = OpenAIChatCompletionChunk.model_construct(
        id="test-id",
        choices=[
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": "model_context_window_exceeded",
            }
        ],
        created=1234567890,
        model="glm-5",
        object="chat.completion.chunk",
    )
    result = BaseOpenAIProvider._convert_completion_chunk_response(openai_chunk)
    assert result.choices[0].finish_reason == "length"


def test_convert_chat_completion_maps_model_context_window_exceeded_to_length() -> None:
    """The non-streaming path shares the normalizer, so it must map the same finish reason."""
    openai_response = OpenAIChatCompletion.model_construct(
        id="test-id",
        choices=[
            {
                "index": 0,
                "finish_reason": "model_context_window_exceeded",
                "message": {"role": "assistant", "content": "partial"},
            }
        ],
        created=1234567890,
        model="glm-5",
        object="chat.completion",
    )
    result = _convert_chat_completion(openai_response)
    assert result.choices[0].finish_reason == "length"


@pytest.mark.parametrize("finish_reason", ["stop", "length", "tool_calls", "content_filter", "function_call"])
def test_convert_chat_completion_preserves_standard_finish_reasons(finish_reason: str) -> None:
    """Standard finish reasons must pass through the normalizer untouched."""
    openai_response = OpenAIChatCompletion.model_construct(
        id="test-id",
        choices=[
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": "Hello"},
            }
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )
    result = _convert_chat_completion(openai_response)
    assert result.choices[0].finish_reason == finish_reason


def test_convert_chat_completion_still_rejects_unknown_finish_reason() -> None:
    """Unmapped values keep failing loudly rather than being coerced into a wrong stop reason."""
    openai_response = OpenAIChatCompletion.model_construct(
        id="test-id",
        choices=[
            {
                "index": 0,
                "finish_reason": "some_unknown_reason",
                "message": {"role": "assistant", "content": "Hello"},
            }
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )
    with pytest.raises(ValidationError):
        _convert_chat_completion(openai_response)


def test_normalize_skips_choice_entries_that_are_not_dicts() -> None:
    """A malformed entry in choices must be stepped over, not crash the normalizer or stop later entries."""
    response_dict = {
        "choices": [
            None,
            "unexpected",
            {
                "index": 0,
                "message": {"role": "assistant", "content": "partial"},
                "finish_reason": "model_context_window_exceeded",
            },
        ]
    }

    result = _normalize_openai_dict_response(response_dict)

    assert result["choices"][0] is None
    assert result["choices"][1] == "unexpected"
    assert result["choices"][2]["finish_reason"] == "length"
