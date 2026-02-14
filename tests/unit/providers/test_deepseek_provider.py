import pytest
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from pydantic import BaseModel

from any_llm.providers.deepseek.deepseek import DeepseekProvider
from any_llm.providers.deepseek.utils import _preprocess_messages
from any_llm.types.completion import CompletionParams


class PersonResponseFormat(BaseModel):
    name: str
    age: int


@pytest.mark.asyncio
async def test_preprocess_messages_with_pydantic_model() -> None:
    """Test that Pydantic model is converted to DeepSeek JSON format."""
    messages = [{"role": "user", "content": "Generate a person"}]
    params = CompletionParams(
        model_id="deepseek-chat",
        messages=messages,
        response_format=PersonResponseFormat,
    )

    processed_params = _preprocess_messages(params)

    assert processed_params.response_format == {"type": "json_object"}

    # Should modify the user message to include JSON schema instructions
    assert len(processed_params.messages) == 1
    assert processed_params.messages[0]["role"] == "user"
    assert "JSON object" in processed_params.messages[0]["content"]
    assert "Generate a person" in processed_params.messages[0]["content"]


@pytest.mark.asyncio
async def test_preprocess_messages_without_response_format() -> None:
    """Test that messages are passed through unchanged when no response_format."""
    messages = [{"role": "user", "content": "Hello"}]
    params = CompletionParams(
        model_id="deepseek-chat",
        messages=messages,
        response_format=None,
    )

    processed_params = _preprocess_messages(params)

    assert processed_params.response_format is None
    assert processed_params.messages == messages


@pytest.mark.asyncio
async def test_preprocess_messages_with_non_pydantic_response_format() -> None:
    """Test that non-Pydantic response_format is passed through unchanged."""
    messages = [{"role": "user", "content": "Hello"}]
    response_format = {"type": "json_object"}
    params = CompletionParams(
        model_id="deepseek-chat",
        messages=messages,
        response_format=response_format,
    )

    processed_params = _preprocess_messages(params)
    assert processed_params.response_format == response_format
    assert processed_params.messages == messages


def test_convert_completion_response_extracts_cached_tokens() -> None:
    """Test that prompt_cache_hit_tokens is extracted into prompt_tokens_details."""
    response = OpenAIChatCompletion.model_validate(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_cache_hit_tokens": 80,
                "prompt_cache_miss_tokens": 20,
            },
        }
    )

    result = DeepseekProvider._convert_completion_response(response)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 150
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 80


def test_convert_completion_response_without_cached_tokens() -> None:
    """Test that prompt_tokens_details is None when no cache tokens are present."""
    response = OpenAIChatCompletion.model_validate(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
    )

    result = DeepseekProvider._convert_completion_response(response)

    assert result.usage is not None
    assert result.usage.prompt_tokens_details is None


def test_convert_chunk_response_extracts_cached_tokens() -> None:
    """Test that streaming chunks extract prompt_cache_hit_tokens into prompt_tokens_details."""
    chunk = OpenAIChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "deepseek-chat",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "prompt_cache_hit_tokens": 80,
                "prompt_cache_miss_tokens": 20,
            },
        }
    )

    result = DeepseekProvider._convert_completion_chunk_response(chunk)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 80
