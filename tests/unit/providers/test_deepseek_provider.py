import dataclasses
from typing import Any

import pytest
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from pydantic import BaseModel

from any_llm.providers.deepseek.deepseek import DeepseekProvider
from any_llm.providers.deepseek.utils import _preprocess_messages, _reinject_reasoning_content
from any_llm.types.completion import CompletionParams


class PersonResponseFormat(BaseModel):
    name: str
    age: int


@dataclasses.dataclass
class PersonDataclass:
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


@pytest.mark.asyncio
async def test_preprocess_messages_with_dataclass() -> None:
    """Test that a dataclass is converted to DeepSeek JSON format."""
    messages = [{"role": "user", "content": "Generate a person"}]
    params = CompletionParams(
        model_id="deepseek-chat",
        messages=messages,
        response_format=PersonDataclass,
    )

    processed_params = _preprocess_messages(params)

    assert processed_params.response_format == {"type": "json_object"}

    assert len(processed_params.messages) == 1
    assert processed_params.messages[0]["role"] == "user"
    assert "JSON object" in processed_params.messages[0]["content"]
    assert "Generate a person" in processed_params.messages[0]["content"]
    assert "name" in processed_params.messages[0]["content"]
    assert "age" in processed_params.messages[0]["content"]


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


def test_convert_chunk_response_without_cached_tokens() -> None:
    """Test that prompt_tokens_details is None for streaming chunks when no cache tokens are present."""
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
            },
        }
    )

    result = DeepseekProvider._convert_completion_chunk_response(chunk)

    assert result.usage is not None
    assert result.usage.prompt_tokens_details is None


def test_deepseek_remaps_max_tokens_back_to_max_tokens() -> None:
    """max_tokens → base remaps to max_completion_tokens → DeepSeek remaps back to max_tokens."""
    params = CompletionParams(model_id="deepseek-chat", messages=[{"role": "user", "content": "hi"}], max_tokens=8192)
    result = DeepseekProvider._convert_completion_params(params)
    assert result["max_tokens"] == 8192
    assert "max_completion_tokens" not in result


def test_deepseek_remaps_max_completion_tokens_to_max_tokens() -> None:
    """max_completion_tokens set directly → DeepSeek remaps to max_tokens."""
    params = CompletionParams(
        model_id="deepseek-chat",
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=4096,
    )
    result = DeepseekProvider._convert_completion_params(params)
    assert result["max_tokens"] == 4096
    assert "max_completion_tokens" not in result


def test_deepseek_no_max_tokens_when_neither_set() -> None:
    """Neither max_tokens nor max_completion_tokens set → neither appears."""
    params = CompletionParams(model_id="deepseek-chat", messages=[{"role": "user", "content": "hi"}])
    result = DeepseekProvider._convert_completion_params(params)
    assert "max_tokens" not in result
    assert "max_completion_tokens" not in result


def test_deepseek_thinking_disabled_by_default_for_v4_model() -> None:
    """V4 model ids without reasoning_effort should default to thinking disabled."""
    params = CompletionParams(
        model_id="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=None,
    )
    result = DeepseekProvider._convert_completion_params(params)
    assert result["extra_body"]["thinking"] == {"type": "disabled"}


def test_deepseek_thinking_disabled_for_none_reasoning_effort_value() -> None:
    """reasoning_effort="none" is also treated as "no reasoning requested"."""
    params = CompletionParams(
        model_id="deepseek-v4-pro",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="none",
    )
    result = DeepseekProvider._convert_completion_params(params)
    assert result["extra_body"]["thinking"] == {"type": "disabled"}


def test_deepseek_thinking_enabled_when_reasoning_effort_set() -> None:
    """An explicit reasoning_effort should enable thinking mode and be passed through."""
    params = CompletionParams(
        model_id="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="high",
    )
    result = DeepseekProvider._convert_completion_params(params)
    assert result["extra_body"]["thinking"] == {"type": "enabled"}
    assert result["reasoning_effort"] == "high"


def test_deepseek_thinking_untouched_for_legacy_model_ids() -> None:
    """Legacy deepseek-chat/deepseek-reasoner ids must not get the thinking toggle injected."""
    for model_id in ("deepseek-chat", "deepseek-reasoner"):
        params = CompletionParams(
            model_id=model_id,
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort="high",
        )
        result = DeepseekProvider._convert_completion_params(params)
        assert "extra_body" not in result


def test_deepseek_thinking_respects_explicit_extra_body_override() -> None:
    """A caller-supplied extra_body/thinking value should not be clobbered by the default."""
    params = CompletionParams(
        model_id="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=None,
    )
    result = DeepseekProvider._convert_completion_params(params, extra_body={"thinking": {"type": "enabled"}})
    assert result["extra_body"]["thinking"] == {"type": "enabled"}


def test_convert_completion_response_stashes_reasoning_into_extra_content() -> None:
    """reasoning_content on the raw response should be mirrored into message.extra_content."""
    response = OpenAIChatCompletion.model_validate(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello!",
                        "reasoning_content": "Thinking about hello...",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }
    )

    result = DeepseekProvider._convert_completion_response(response)

    message = result.choices[0].message
    assert message.reasoning is not None
    assert message.reasoning.content == "Thinking about hello..."
    assert message.extra_content == {"deepseek": {"reasoning_content": "Thinking about hello..."}}


def test_convert_completion_response_no_extra_content_without_reasoning() -> None:
    """No reasoning present → extra_content should not be populated."""
    response = OpenAIChatCompletion.model_validate(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }
    )

    result = DeepseekProvider._convert_completion_response(response)

    assert result.choices[0].message.extra_content is None


def test_reinject_reasoning_content_on_tool_call_message() -> None:
    """A replayed assistant message with tool_calls should get reasoning_content restored."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ],
            "extra_content": {"deepseek": {"reasoning_content": "I should call get_weather."}},
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
    ]

    result = _reinject_reasoning_content(messages)

    assert result[1]["reasoning_content"] == "I should call get_weather."
    # The any_llm-internal extra_content must not be forwarded to DeepSeek's API.
    assert "extra_content" not in result[1]
    # Original message dict must not be mutated in place.
    assert "reasoning_content" not in messages[1]
    assert "extra_content" in messages[1]


def test_reinject_reasoning_content_skips_non_tool_call_message() -> None:
    """Assistant messages without tool_calls should be left untouched."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello",
            "extra_content": {"deepseek": {"reasoning_content": "greeting"}},
        },
    ]

    result = _reinject_reasoning_content(messages)

    assert "reasoning_content" not in result[1]
    # extra_content is any_llm-internal and is stripped even when reasoning is not reinjected.
    assert "extra_content" not in result[1]


def test_reinject_reasoning_content_handles_missing_extra_content() -> None:
    """A tool-call message with no extra_content should pass through unchanged."""
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ],
        },
    ]

    result = _reinject_reasoning_content(messages)

    assert "reasoning_content" not in result[0]
