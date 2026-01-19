from unittest.mock import Mock

import pytest

from any_llm.providers.llama.llama import LlamaProvider
from any_llm.providers.llama.utils import (
    _convert_tool_call_arguments,
    _extract_content_text,
    _map_stop_reason_to_finish_reason,
    _patch_json_schema,
)


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
