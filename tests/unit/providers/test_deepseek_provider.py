import dataclasses
from typing import Any

import pytest
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from pydantic import BaseModel

from any_llm.exceptions import InvalidRequestError
from any_llm.providers.deepseek.deepseek import DeepseekProvider
from any_llm.providers.deepseek.utils import _preprocess_messages, _reinject_reasoning_content
from any_llm.types.completion import CompletionParams, ReasoningEffort


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


def test_deepseek_preserves_provider_thinking_default() -> None:
    """No explicit effort leaves DeepSeek's enabled/high provider default in control."""
    params = CompletionParams(
        model_id="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=None,
    )
    result = DeepseekProvider._convert_completion_params(params)
    assert "reasoning_effort" not in result
    assert "extra_body" not in result


def test_deepseek_thinking_disabled_for_none_reasoning_effort_value() -> None:
    """An explicit none uses DeepSeek's thinking toggle, not an invalid wire effort."""
    params = CompletionParams(
        model_id="deepseek-v4-pro",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="none",
    )
    result = DeepseekProvider._convert_completion_params(params)
    assert result["extra_body"]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in result


def test_deepseek_auto_preserves_provider_thinking_default() -> None:
    """The normalized auto sentinel does not override DeepSeek's provider default."""
    params = CompletionParams(
        model_id="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="auto",
    )
    result = DeepseekProvider._convert_completion_params(params)
    assert "reasoning_effort" not in result
    assert "extra_body" not in result


@pytest.mark.parametrize(
    ("reasoning_effort", "expected_effort"),
    [("low", "low"), ("medium", "high"), ("high", "high"), ("xhigh", "high"), ("max", "max")],
)
def test_deepseek_maps_current_reasoning_efforts(reasoning_effort: ReasoningEffort, expected_effort: str) -> None:
    """Normalized efforts map to DeepSeek's current low, high, and max wire values."""
    params = CompletionParams(
        model_id="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort=reasoning_effort,
    )
    result = DeepseekProvider._convert_completion_params(params)
    assert result["extra_body"]["thinking"] == {"type": "enabled"}
    assert result["reasoning_effort"] == expected_effort


def test_deepseek_rejects_unsupported_minimal_reasoning_effort() -> None:
    """DeepSeek Chat does not document OpenAI's minimal effort."""
    params = CompletionParams(
        model_id="deepseek-v4-pro",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="minimal",
    )

    with pytest.raises(InvalidRequestError, match="minimal"):
        DeepseekProvider._convert_completion_params(params)


def test_deepseek_thinking_respects_explicit_extra_body_override() -> None:
    """Caller-supplied DeepSeek fields take precedence over normalized controls."""
    params = CompletionParams(
        model_id="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="max",
        user="normalized-user",
    )
    result = DeepseekProvider._convert_completion_params(
        params,
        extra_body={"thinking": {"type": "disabled"}, "user_id": "caller-user"},
    )
    assert result["reasoning_effort"] == "max"
    assert result["extra_body"] == {"thinking": {"type": "disabled"}, "user_id": "caller-user"}


def test_deepseek_maps_user_id_and_omits_unsupported_fields() -> None:
    params = CompletionParams(
        model_id="deepseek-v4-flash",
        messages=[{"role": "user", "content": "hi"}],
        user="account_42",
        n=2,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        seed=7,
        parallel_tool_calls=False,
        logit_bias={"123": 1.0},
        prompt_cache_key="cache-key",
        service_tier="priority",
    )

    result = DeepseekProvider._convert_completion_params(params)

    assert result["extra_body"]["user_id"] == "account_42"
    for field in (
        "user",
        "n",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "parallel_tool_calls",
        "logit_bias",
        "prompt_cache_key",
        "service_tier",
    ):
        assert field not in result


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

    result = _reinject_reasoning_content(messages, replay_reasoning=True)

    assert result[1]["reasoning_content"] == "I should call get_weather."
    # The any_llm-internal extra_content must not be forwarded to DeepSeek's API.
    assert "extra_content" not in result[1]
    # Original message dict must not be mutated in place.
    assert "reasoning_content" not in messages[1]
    assert "extra_content" in messages[1]


def test_reinject_reasoning_content_includes_assistant_turn_without_tool_call() -> None:
    """A request carrying tools replays reasoning from every previous assistant turn."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello",
            "extra_content": {"deepseek": {"reasoning_content": "greeting"}},
        },
    ]

    result = _reinject_reasoning_content(messages, replay_reasoning=True)

    assert result[1]["reasoning_content"] == "greeting"
    assert "extra_content" not in result[1]


def test_preprocess_messages_replays_all_assistant_reasoning_when_tools_are_present() -> None:
    params = CompletionParams(
        model_id="deepseek-v4-pro",
        messages=[
            {
                "role": "assistant",
                "content": "hello",
                "extra_content": {"deepseek": {"reasoning_content": "greeting"}},
            }
        ],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )

    processed = _preprocess_messages(params)

    assert processed.messages[0]["reasoning_content"] == "greeting"
    assert "extra_content" not in processed.messages[0]


def test_reinject_reasoning_content_omits_reasoning_without_tools() -> None:
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "hello",
            "extra_content": {"deepseek": {"reasoning_content": "greeting"}},
        }
    ]

    result = _reinject_reasoning_content(messages, replay_reasoning=False)

    assert "reasoning_content" not in result[0]
    assert "extra_content" not in result[0]


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

    result = _reinject_reasoning_content(messages, replay_reasoning=True)

    assert "reasoning_content" not in result[0]
