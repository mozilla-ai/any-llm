from unittest.mock import Mock

import pytest

from any_llm.providers.llama.llama import LlamaProvider
from any_llm.providers.llama.utils import (
    _convert_tool_call_arguments,
    _extract_content_text,
    _map_stop_reason_to_finish_reason,
    _patch_json_schema,
)


class TestMapStopReasonToFinishReason:
    def test_end_of_turn_maps_to_stop(self) -> None:
        assert _map_stop_reason_to_finish_reason("end_of_turn") == "stop"

    def test_tool_call_maps_to_tool_calls(self) -> None:
        assert _map_stop_reason_to_finish_reason("tool_call") == "tool_calls"

    def test_max_tokens_maps_to_length(self) -> None:
        assert _map_stop_reason_to_finish_reason("max_tokens") == "length"

    def test_none_returns_none(self) -> None:
        assert _map_stop_reason_to_finish_reason(None) is None

    def test_unknown_reason_defaults_to_stop(self) -> None:
        assert _map_stop_reason_to_finish_reason("unknown_reason") == "stop"


class TestExtractContentText:
    def test_extracts_string_content(self) -> None:
        assert _extract_content_text("Hello, world!") == "Hello, world!"

    def test_extracts_from_text_attribute(self) -> None:
        mock_content = Mock()
        mock_content.text = "Content from text attribute"
        assert _extract_content_text(mock_content) == "Content from text attribute"

    def test_extracts_from_dict(self) -> None:
        content = {"text": "Content from dict"}
        assert _extract_content_text(content) == "Content from dict"

    def test_returns_none_for_none(self) -> None:
        assert _extract_content_text(None) is None

    def test_converts_unknown_type_to_string(self) -> None:
        mock_content = Mock(spec=[])  # No text attribute
        mock_content.__str__ = Mock(return_value="stringified")
        result = _extract_content_text(mock_content)
        assert result == "stringified"


class TestConvertToolCallArguments:
    def test_converts_dict_to_json_string(self) -> None:
        args = {"param1": "value1", "param2": 42}
        result = _convert_tool_call_arguments(args)
        assert result == '{"param1": "value1", "param2": 42}'

    def test_returns_string_as_is(self) -> None:
        args = '{"already": "json"}'
        assert _convert_tool_call_arguments(args) == '{"already": "json"}'

    def test_returns_empty_object_for_none(self) -> None:
        assert _convert_tool_call_arguments(None) == "{}"


class TestPatchJsonSchema:
    def test_patches_oneof_without_type(self) -> None:
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

    def test_does_not_modify_oneof_with_existing_type(self) -> None:
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


class TestConvertCompletionResponse:
    def test_converts_basic_response(self) -> None:
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

    def test_converts_response_with_tool_calls(self) -> None:
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

    def test_extracts_usage_from_metrics(self) -> None:
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


class TestConvertCompletionChunkResponse:
    def test_converts_text_chunk(self) -> None:
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

    def test_converts_chunk_with_stop_reason(self) -> None:
        mock_chunk = Mock()
        mock_event = Mock()
        mock_delta = Mock()

        mock_delta.text = None
        mock_delta.role = None
        mock_delta.tool_call = None
        mock_event.delta = mock_delta
        mock_event.stop_reason = "end_of_turn"
        mock_chunk.event = mock_event

        result = LlamaProvider._convert_completion_chunk_response(mock_chunk, model="llama-model")

        assert result.choices[0].finish_reason == "stop"

    def test_converts_chunk_with_tool_call(self) -> None:
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
        assert result.choices[0].delta.tool_calls[0]["function"]["name"] == "search"

    def test_handles_chunk_without_event(self) -> None:
        mock_chunk = Mock()
        mock_chunk.event = None

        result = LlamaProvider._convert_completion_chunk_response(mock_chunk, model="llama-model")

        assert result.choices[0].delta.content is None
        assert result.choices[0].finish_reason is None
