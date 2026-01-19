from unittest.mock import AsyncMock, Mock, patch

import cmath

import pytest
from pydantic import BaseModel

from any_llm.providers.llama.llama import LlamaProvider
from any_llm.providers.llama.utils import (
    _convert_tool_call_arguments,
    _extract_content_text,
    _map_stop_reason_to_finish_reason,
    _patch_json_schema,
)
from any_llm.types.completion import CompletionParams


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_of_turn", "stop"),
        ("tool_call", "tool_calls"),
        ("max_tokens", "length"),
        ("unknown_reason", "stop"),
        (None, None),
    ],
)
def test_maps_stop_reason_correctly(stop_reason: str | None, expected: str | None) -> None:
    assert _map_stop_reason_to_finish_reason(stop_reason) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("Hello, world!", "Hello, world!"),
        ({"text": "Content from dict"}, "Content from dict"),
        (None, None),
    ],
)
def test_extracts_content(content: str | dict | None, expected: str | None) -> None:
    assert _extract_content_text(content) == expected

def test_extracts_from_text_attribute() -> None:
    mock_content = Mock()
    mock_content.text = "Content from text attribute"
    assert _extract_content_text(mock_content) == "Content from text attribute"

def test_converts_unknown_type_to_string() -> None:
    class CustomContent:
        """A content type without a text attribute."""

        def __str__(self) -> str:
            return "stringified"

    result = _extract_content_text(CustomContent())
    assert result == "stringified"


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"param1": "value1", "param2": 42}, '{"param1": "value1", "param2": 42}'),
        ('{"already": "json"}', '{"already": "json"}'),
        (None, "{}"),
    ],
)
def test_converts_arguments(arguments: dict | str | None, expected: str) -> None:
    assert _convert_tool_call_arguments(arguments) == expected


def test_patches_oneof_without_type() -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "test_func",
            "parameters": {
                "type": "object",
                "properties": {
                    "union_param": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                },
            },
        },
    }
    result = _patch_json_schema(schema)
    assert result["function"]["parameters"]["properties"]["union_param"]["type"] == "string"

def test_does_not_modify_oneof_with_existing_type() -> None:
    schema = {
        "type": "function",
        "function": {
            "name": "test_func",
            "parameters": {
                "type": "object",
                "properties": {
                    "union_param": {"oneOf": [{"type": "string"}], "type": "number"},
                },
            },
        },
    }
    result = _patch_json_schema(schema)
    assert result["function"]["parameters"]["properties"]["union_param"]["type"] == "number"


def test_converts_basic_response() -> None:
    mock_response = Mock()
    mock_message = Mock()
    mock_message.role = "assistant"
    mock_message.content = "Hello!"
    mock_message.tool_calls = None

    mock_response.completion_message = mock_message
    mock_response.stop_reason = "end_of_turn"
    mock_response.metrics = None

    result = LlamaProvider._convert_completion_response(mock_response, model_id="llama-model")

    assert result.model == "llama-model"
    assert len(result.choices) == 1
    assert result.choices[0].message.content == "Hello!"
    assert result.choices[0].message.role == "assistant"
    assert result.choices[0].finish_reason == "stop"

def test_converts_response_with_tool_calls() -> None:
    mock_response = Mock()
    mock_message = Mock()
    mock_message.role = "assistant"
    mock_message.content = None

    mock_tool_call = Mock()
    mock_tool_call.id = "call_123"
    mock_tool_call.tool_name = "get_weather"
    mock_tool_call.name = None
    mock_tool_call.call_id = None
    mock_tool_call.arguments = {"location": "San Francisco"}

    mock_message.tool_calls = [mock_tool_call]
    mock_response.completion_message = mock_message
    mock_response.stop_reason = "tool_call"
    mock_response.metrics = None

    result = LlamaProvider._convert_completion_response(mock_response, model_id="llama-model")

    assert result.choices[0].finish_reason == "tool_calls"
    assert result.choices[0].message.tool_calls is not None
    assert len(result.choices[0].message.tool_calls) == 1
    assert result.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert result.choices[0].message.tool_calls[0].function.arguments == '{"location": "San Francisco"}'

def test_extracts_usage_from_metrics() -> None:
    mock_response = Mock()
    mock_message = Mock()
    mock_message.role = "assistant"
    mock_message.content = "Response"
    mock_message.tool_calls = None

    mock_metrics = Mock()
    mock_metrics.prompt_token_count = 10
    mock_metrics.completion_token_count = 5

    mock_response.completion_message = mock_message
    mock_response.stop_reason = "end_of_turn"
    mock_response.metrics = mock_metrics

    result = LlamaProvider._convert_completion_response(mock_response, model_id="llama-model")

    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15


def test_converts_text_chunk() -> None:
    mock_chunk = Mock()
    mock_event = Mock()
    mock_delta = Mock()

    mock_delta.text = "Hello"
    mock_delta.role = None
    mock_delta.tool_call = None
    mock_event.delta = mock_delta
    mock_event.stop_reason = None
    mock_chunk.event = mock_event

    result = LlamaProvider._convert_completion_chunk_response(mock_chunk, model="llama-model")

    assert result.choices[0].delta.content == "Hello"
    assert result.choices[0].finish_reason is None
    assert result.model == "llama-model"

@pytest.mark.parametrize(
    ("stop_reason", "expected_finish_reason"),
    [
        ("end_of_turn", "stop"),
        ("tool_call", "tool_calls"),
        ("max_tokens", "length"),
    ],
)
def test_converts_chunk_with_stop_reason(stop_reason: str, expected_finish_reason: str) -> None:
    mock_chunk = Mock()
    mock_event = Mock()
    mock_delta = Mock()

    mock_delta.text = None
    mock_delta.role = None
    mock_delta.tool_call = None
    mock_event.delta = mock_delta
    mock_event.stop_reason = stop_reason
    mock_chunk.event = mock_event

    result = LlamaProvider._convert_completion_chunk_response(mock_chunk, model="llama-model")

    assert result.choices[0].finish_reason == expected_finish_reason

def test_converts_chunk_with_tool_call() -> None:
    mock_chunk = Mock()
    mock_event = Mock()
    mock_delta = Mock()
    mock_tool_call = Mock()

    mock_tool_call.index = 0
    mock_tool_call.id = "call_456"
    mock_tool_call.tool_name = "search"
    mock_tool_call.arguments = '{"query": "weather"}'

    mock_delta.text = None
    mock_delta.role = None
    mock_delta.tool_call = mock_tool_call
    mock_event.delta = mock_delta
    mock_event.stop_reason = None
    mock_chunk.event = mock_event

    result = LlamaProvider._convert_completion_chunk_response(mock_chunk, model="llama-model")

    assert result.choices[0].delta.tool_calls is not None
    assert len(result.choices[0].delta.tool_calls) == 1
    # tool_calls are validated into ChoiceDeltaToolCall objects
    assert result.choices[0].delta.tool_calls[0].function.name == "search"

def test_handles_chunk_without_event() -> None:
    mock_chunk = Mock()
    mock_chunk.event = None

    result = LlamaProvider._convert_completion_chunk_response(mock_chunk, model="llama-model")

    assert result.choices[0].delta.content is None
    assert result.choices[0].finish_reason is None


def test_converts_basic_params() -> None:
    params = CompletionParams(
        model_id="llama-model",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7,
        max_tokens=100,
    )
    result = LlamaProvider._convert_completion_params(params)

    assert cmath.isclose(result["temperature"], 0.7)
    assert result["max_tokens"] == 100
    assert "model_id" not in result
    assert "messages" not in result
    assert "stream" not in result

def test_excludes_none_values() -> None:
    params = CompletionParams(
        model_id="llama-model",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=None,
        max_tokens=None,
    )
    result = LlamaProvider._convert_completion_params(params)

    assert "temperature" not in result
    assert "max_tokens" not in result

def test_excludes_reasoning_effort() -> None:
    params = CompletionParams(
        model_id="llama-model",
        messages=[{"role": "user", "content": "Hello"}],
        reasoning_effort="high",
    )
    result = LlamaProvider._convert_completion_params(params)

    assert "reasoning_effort" not in result

def test_patches_tools_json_schema() -> None:
    params = CompletionParams(
        model_id="llama-model",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "test_func",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "union_param": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
                        },
                    },
                },
            }
        ],
    )
    result = LlamaProvider._convert_completion_params(params)

    assert result["tools"][0]["function"]["parameters"]["properties"]["union_param"]["type"] == "string"

def test_does_not_patch_non_function_tools() -> None:
    params = CompletionParams(
        model_id="llama-model",
        messages=[{"role": "user", "content": "Hello"}],
        tools=[{"type": "other", "data": "value"}],
    )
    result = LlamaProvider._convert_completion_params(params)

    assert result["tools"][0] == {"type": "other", "data": "value"}

def test_converts_pydantic_response_format() -> None:
    class ResponseSchema(BaseModel):
        name: str
        value: int

    params = CompletionParams(
        model_id="llama-model",
        messages=[{"role": "user", "content": "Hello"}],
        response_format=ResponseSchema,
    )
    result = LlamaProvider._convert_completion_params(params)

    assert result["response_format"]["type"] == "json_schema"
    assert result["response_format"]["json_schema"]["name"] == "response_schema"
    assert result["response_format"]["json_schema"]["strict"] is True
    assert "properties" in result["response_format"]["json_schema"]["schema"]

def test_merges_additional_kwargs() -> None:
    params = CompletionParams(
        model_id="llama-model",
        messages=[{"role": "user", "content": "Hello"}],
    )
    result = LlamaProvider._convert_completion_params(params, custom_param="custom_value")

    assert result["custom_param"] == "custom_value"


def test_returns_none_for_empty_list() -> None:
    assert LlamaProvider._build_tool_calls([]) is None

def test_returns_none_for_none() -> None:
    assert LlamaProvider._build_tool_calls(None) is None

def test_uses_call_id_fallback_when_id_is_none() -> None:
    mock_tool_call = Mock()
    mock_tool_call.id = None
    mock_tool_call.call_id = "fallback_call_id"
    mock_tool_call.tool_name = "test_tool"
    mock_tool_call.name = None
    mock_tool_call.arguments = {}

    result = LlamaProvider._build_tool_calls([mock_tool_call])

    assert result is not None
    assert result[0].id == "fallback_call_id"

def test_uses_name_fallback_when_tool_name_is_none() -> None:
    mock_tool_call = Mock()
    mock_tool_call.id = "call_123"
    mock_tool_call.call_id = None
    mock_tool_call.tool_name = None
    mock_tool_call.name = "fallback_name"
    mock_tool_call.arguments = {}

    result = LlamaProvider._build_tool_calls([mock_tool_call])

    assert result is not None
    assert result[0].function.name == "fallback_name"

def test_handles_multiple_tool_calls() -> None:
    tool_calls = []
    for i in range(3):
        tc = Mock()
        tc.id = f"call_{i}"
        tc.call_id = None
        tc.tool_name = f"tool_{i}"
        tc.name = None
        tc.arguments = {"arg": i}
        tool_calls.append(tc)

    result = LlamaProvider._build_tool_calls(tool_calls)

    assert result is not None
    assert len(result) == 3
    for i, tc in enumerate(result):
        assert tc.id == f"call_{i}"
        assert tc.function.name == f"tool_{i}"


def test_returns_none_for_no_metrics() -> None:
    assert LlamaProvider._build_usage(None) is None

def test_handles_none_token_counts() -> None:
    mock_metrics = Mock()
    mock_metrics.prompt_token_count = None
    mock_metrics.completion_token_count = None

    result = LlamaProvider._build_usage(mock_metrics)

    assert result is not None
    assert result.prompt_tokens == 0
    assert result.completion_tokens == 0
    assert result.total_tokens == 0

def test_calculates_total_tokens() -> None:
    mock_metrics = Mock()
    mock_metrics.prompt_token_count = 50
    mock_metrics.completion_token_count = 25

    result = LlamaProvider._build_usage(mock_metrics)

    assert result is not None
    assert result.prompt_tokens == 50
    assert result.completion_tokens == 25
    assert result.total_tokens == 75


def test_handles_none_index() -> None:
    mock_tool_call = Mock()
    mock_tool_call.index = None
    mock_tool_call.id = "call_123"
    mock_tool_call.tool_name = "test_tool"
    mock_tool_call.arguments = '{"key": "value"}'

    result = LlamaProvider._build_streaming_tool_call(mock_tool_call)

    assert result["index"] == 0

def test_handles_tool_call_without_name_or_arguments() -> None:
    mock_tool_call = Mock()
    mock_tool_call.index = 0
    mock_tool_call.id = "call_123"
    mock_tool_call.tool_name = None
    mock_tool_call.arguments = None

    result = LlamaProvider._build_streaming_tool_call(mock_tool_call)

    assert result["index"] == 0
    assert result["id"] == "call_123"
    assert result["type"] == "function"
    assert "function" not in result

def test_includes_function_when_name_present() -> None:
    mock_tool_call = Mock()
    mock_tool_call.index = 1
    mock_tool_call.id = "call_456"
    mock_tool_call.tool_name = "get_weather"
    mock_tool_call.arguments = None

    result = LlamaProvider._build_streaming_tool_call(mock_tool_call)

    assert "function" in result
    assert result["function"]["name"] == "get_weather"

def test_includes_function_when_arguments_present() -> None:
    mock_tool_call = Mock()
    mock_tool_call.index = 0
    mock_tool_call.id = "call_789"
    mock_tool_call.tool_name = None
    mock_tool_call.arguments = {"location": "NYC"}

    result = LlamaProvider._build_streaming_tool_call(mock_tool_call)

    assert "function" in result
    assert result["function"]["arguments"] == '{"location": "NYC"}'


def test_extracts_role_from_delta() -> None:
    mock_response = Mock()
    mock_event = Mock()
    mock_delta = Mock()

    mock_delta.text = None
    mock_delta.role = "assistant"
    mock_delta.tool_call = None
    mock_event.delta = mock_delta
    mock_event.stop_reason = None
    mock_response.event = mock_event

    delta, finish_reason = LlamaProvider._extract_chunk_delta(mock_response)

    assert delta["role"] == "assistant"
    assert finish_reason is None

def test_handles_missing_delta() -> None:
    mock_response = Mock()
    mock_event = Mock()
    mock_event.delta = None
    mock_event.stop_reason = "end_of_turn"
    mock_response.event = mock_event

    delta, finish_reason = LlamaProvider._extract_chunk_delta(mock_response)

    assert delta == {}
    assert finish_reason == "stop"


def test_handles_response_without_completion_message() -> None:
    mock_response = Mock()
    mock_response.completion_message = None
    mock_response.metrics = None

    result = LlamaProvider._convert_completion_response(mock_response, model_id="llama-model")

    assert result.model == "llama-model"
    assert len(result.choices) == 0

def test_handles_missing_stop_reason() -> None:
    mock_response = Mock()
    mock_message = Mock()
    mock_message.role = "assistant"
    mock_message.content = "Hello"
    mock_message.tool_calls = None

    mock_response.completion_message = mock_message
    mock_response.stop_reason = None
    mock_response.metrics = None

    result = LlamaProvider._convert_completion_response(mock_response, model_id="llama-model")

    assert result.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_acompletion_non_streaming() -> None:
    with patch.object(LlamaProvider, "_init_client"):
        provider = LlamaProvider(api_key="test-key")
        provider.client = Mock()

        mock_response = Mock()
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Hello!"
        mock_message.tool_calls = None
        mock_response.completion_message = mock_message
        mock_response.stop_reason = "end_of_turn"
        mock_response.metrics = None

        provider.client.chat = Mock()
        provider.client.chat.completions = Mock()
        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        params = CompletionParams(
            model_id="llama-model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )

        result = await provider._acompletion(params)

        assert result.choices[0].message.content == "Hello!"
        provider.client.chat.completions.create.assert_called_once()

@pytest.mark.asyncio
async def test_acompletion_streaming_returns_async_iterator() -> None:
    with patch.object(LlamaProvider, "_init_client"):
        provider = LlamaProvider(api_key="test-key")
        provider.client = Mock()
        provider.client.chat = Mock()
        provider.client.chat.completions = Mock()

        async def mock_stream():
            mock_chunk = Mock()
            mock_event = Mock()
            mock_delta = Mock()
            mock_delta.text = "Hello"
            mock_delta.role = None
            mock_delta.tool_call = None
            mock_event.delta = mock_delta
            mock_event.stop_reason = None
            mock_chunk.event = mock_event
            yield mock_chunk

        provider.client.chat.completions.create = AsyncMock(return_value=mock_stream())

        params = CompletionParams(
            model_id="llama-model",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )

        result = await provider._acompletion(params)

        # Result should be an async iterator
        chunks = []
        async for chunk in result:
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Hello"
