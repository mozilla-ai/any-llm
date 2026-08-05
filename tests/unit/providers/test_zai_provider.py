import pytest
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from pydantic import BaseModel, ValidationError

from any_llm.exceptions import ProviderError, UnsupportedParameterError
from any_llm.providers.zai.zai import ZaiProvider
from any_llm.types.completion import CompletionParams


def test_zai_unsupported_response_format() -> None:
    class ResponseFormatModel(BaseModel):
        response: str

    params = CompletionParams(
        model_id="zai-model", messages=[{"role": "user", "content": "Hello"}], response_format=ResponseFormatModel
    )
    with pytest.raises(UnsupportedParameterError, match="'response_format' is not supported for zai"):
        ZaiProvider._convert_completion_params(params)


def test_zai_remaps_max_tokens_to_max_completion_tokens() -> None:
    params = CompletionParams(model_id="zai-model", messages=[{"role": "user", "content": "Hello"}], max_tokens=8192)
    result = ZaiProvider._convert_completion_params(params)
    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 8192


def test_zai_completion_maps_model_context_window_exceeded_to_length() -> None:
    """z.ai's GLM models end a truncated non-streaming response with a finish reason outside the OpenAI literal set."""
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
    result = ZaiProvider._convert_completion_response(openai_response)
    assert result.choices[0].finish_reason == "length"


def test_zai_chunk_maps_model_context_window_exceeded_to_length() -> None:
    """The streaming path shares the pre-pass, so it must map the same finish reason."""
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
    result = ZaiProvider._convert_completion_chunk_response(openai_chunk)
    assert result.choices[0].finish_reason == "length"


def test_zai_completion_maps_sensitive_to_content_filter() -> None:
    """GLM's sensitive stop reason is a safety interception, which maps to OpenAI's content_filter."""
    openai_response = OpenAIChatCompletion.model_construct(
        id="test-id",
        choices=[
            {
                "index": 0,
                "finish_reason": "sensitive",
                "message": {"role": "assistant", "content": "partial"},
            }
        ],
        created=1234567890,
        model="glm-5",
        object="chat.completion",
    )
    result = ZaiProvider._convert_completion_response(openai_response)
    assert result.choices[0].finish_reason == "content_filter"


def test_zai_chunk_maps_sensitive_to_content_filter() -> None:
    """The streaming path maps GLM's sensitive stop reason to content_filter."""
    openai_chunk = OpenAIChatCompletionChunk.model_construct(
        id="test-id",
        choices=[
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": "sensitive",
            }
        ],
        created=1234567890,
        model="glm-5",
        object="chat.completion.chunk",
    )
    result = ZaiProvider._convert_completion_chunk_response(openai_chunk)
    assert result.choices[0].finish_reason == "content_filter"


@pytest.mark.parametrize("finish_reason", ["stop", "length", "tool_calls", "content_filter", "function_call"])
def test_zai_completion_preserves_standard_finish_reasons(finish_reason: str) -> None:
    """Standard OpenAI finish reasons pass through the zai pre-pass untouched."""
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
    result = ZaiProvider._convert_completion_response(openai_response)
    assert result.choices[0].finish_reason == finish_reason


def test_zai_completion_still_rejects_unknown_finish_reason() -> None:
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
        ZaiProvider._convert_completion_response(openai_response)


def test_zai_completion_still_rejects_network_error() -> None:
    """network_error is deliberately unmapped: a transport failure is not a completion reason."""
    openai_response = OpenAIChatCompletion.model_construct(
        id="test-id",
        choices=[
            {
                "index": 0,
                "finish_reason": "network_error",
                "message": {"role": "assistant", "content": "Hello"},
            }
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion",
    )
    with pytest.raises(ValidationError):
        ZaiProvider._convert_completion_response(openai_response)


def test_zai_completion_empty_response_still_raises_provider_error() -> None:
    """A falsy choices list must not crash the pre-pass; the base layer's empty-response ProviderError surfaces."""
    openai_response = OpenAIChatCompletion.model_construct(
        id=None,
        choices=None,
        created=None,
        model=None,
        object="chat.completion",
    )
    with pytest.raises(ProviderError) as exc_info:
        ZaiProvider._convert_completion_response(openai_response)
    assert "Provider returned an empty response" in str(exc_info.value)
