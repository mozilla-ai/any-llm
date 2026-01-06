from datetime import datetime
from unittest.mock import Mock

from together.types.chat_completions import ChatCompletionChunk as TogetherChatCompletionChunk

from any_llm.providers.together.utils import _create_openai_chunk_from_together_chunk
from any_llm.types.completion import ChatCompletionChunk


def test_create_openai_chunk_handles_empty_choices() -> None:
    together_chunk = Mock(spec=TogetherChatCompletionChunk)
    together_chunk.choices = None
    together_chunk.id = "test-id"
    together_chunk.created = int(datetime.now().timestamp())
    together_chunk.model = "test-model"
    together_chunk.usage = None

    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert isinstance(result, ChatCompletionChunk)
    assert result.choices == []
    assert result.id == "test-id"
    assert result.model == "test-model"

    together_chunk.choices = []
    result = _create_openai_chunk_from_together_chunk(together_chunk)
    assert result.choices == []


def test_create_openai_chunk_handles_missing_delta() -> None:
    """Test that the function handles choices with None delta gracefully."""
    choice_mock = Mock()
    choice_mock.delta = None
    choice_mock.index = 0
    choice_mock.finish_reason = "stop"

    together_chunk = Mock(spec=TogetherChatCompletionChunk)
    together_chunk.choices = [choice_mock]
    together_chunk.id = "test-id"
    together_chunk.created = int(datetime.now().timestamp())
    together_chunk.model = "test-model"
    together_chunk.usage = None

    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert len(result.choices) == 1
    assert result.choices[0].delta.content is None
    assert result.choices[0].delta.role is None


def test_create_openai_chunk_with_tool_calls_dict_format() -> None:
    """Test handling tool calls in dict format."""
    tool_call_dict = {
        "id": "call-123",
        "index": 0,
        "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
    }

    delta_mock = Mock()
    delta_mock.content = None
    delta_mock.role = "assistant"
    delta_mock.reasoning = None
    delta_mock.tool_calls = [tool_call_dict]

    choice_mock = Mock()
    choice_mock.delta = delta_mock
    choice_mock.index = 0
    choice_mock.finish_reason = "tool_calls"

    together_chunk = Mock(spec=TogetherChatCompletionChunk)
    together_chunk.choices = [choice_mock]
    together_chunk.id = "test-id"
    together_chunk.created = int(datetime.now().timestamp())
    together_chunk.model = "test-model"
    together_chunk.usage = None

    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert len(result.choices) == 1
    assert result.choices[0].delta.tool_calls is not None
    assert len(result.choices[0].delta.tool_calls) == 1
    tool_call = result.choices[0].delta.tool_calls[0]
    assert tool_call.id == "call-123"
    assert tool_call.index == 0
    assert tool_call.function is not None
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"location": "Paris"}'


def test_create_openai_chunk_with_tool_calls_object_format() -> None:
    """Test handling tool calls in object format."""
    func_mock = Mock()
    func_mock.name = "search"
    func_mock.arguments = '{"query": "test"}'

    tool_call_mock = Mock()
    tool_call_mock.id = "call-456"
    tool_call_mock.index = 0
    tool_call_mock.function = func_mock

    delta_mock = Mock()
    delta_mock.content = None
    delta_mock.role = "assistant"
    delta_mock.reasoning = None
    delta_mock.tool_calls = [tool_call_mock]

    choice_mock = Mock()
    choice_mock.delta = delta_mock
    choice_mock.index = 0
    choice_mock.finish_reason = "tool_calls"

    together_chunk = Mock(spec=TogetherChatCompletionChunk)
    together_chunk.choices = [choice_mock]
    together_chunk.id = "test-id"
    together_chunk.created = int(datetime.now().timestamp())
    together_chunk.model = "test-model"
    together_chunk.usage = None

    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert result.choices[0].delta.tool_calls is not None
    tool_call = result.choices[0].delta.tool_calls[0]
    assert tool_call.id == "call-456"
    assert tool_call.function is not None
    assert tool_call.function.name == "search"
    assert tool_call.function.arguments == '{"query": "test"}'


def test_create_openai_chunk_with_tool_calls_missing_id() -> None:
    """Test handling tool calls without id generates uuid."""
    tool_call_dict = {
        "index": 0,
        "function": {"name": "func", "arguments": "{}"},
    }

    delta_mock = Mock()
    delta_mock.content = None
    delta_mock.role = "assistant"
    delta_mock.reasoning = None
    delta_mock.tool_calls = [tool_call_dict]

    choice_mock = Mock()
    choice_mock.delta = delta_mock
    choice_mock.index = 0
    choice_mock.finish_reason = "tool_calls"

    together_chunk = Mock(spec=TogetherChatCompletionChunk)
    together_chunk.choices = [choice_mock]
    together_chunk.id = "test-id"
    together_chunk.created = int(datetime.now().timestamp())
    together_chunk.model = "test-model"
    together_chunk.usage = None

    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert result.choices[0].delta.tool_calls is not None
    tool_call = result.choices[0].delta.tool_calls[0]
    assert tool_call.id is not None
    assert len(tool_call.id) > 0


def test_create_openai_chunk_with_tool_calls_object_missing_function() -> None:
    """Test handling tool calls object without function attribute."""
    tool_call_mock = Mock()
    tool_call_mock.id = None
    tool_call_mock.index = 0
    tool_call_mock.function = None

    delta_mock = Mock()
    delta_mock.content = None
    delta_mock.role = "assistant"
    delta_mock.reasoning = None
    delta_mock.tool_calls = [tool_call_mock]

    choice_mock = Mock()
    choice_mock.delta = delta_mock
    choice_mock.index = 0
    choice_mock.finish_reason = "tool_calls"

    together_chunk = Mock(spec=TogetherChatCompletionChunk)
    together_chunk.choices = [choice_mock]
    together_chunk.id = "test-id"
    together_chunk.created = int(datetime.now().timestamp())
    together_chunk.model = "test-model"
    together_chunk.usage = None

    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert result.choices[0].delta.tool_calls is not None
    tool_call = result.choices[0].delta.tool_calls[0]
    assert tool_call.function is not None
    assert tool_call.function.name == ""
    assert tool_call.function.arguments == ""


def test_create_openai_chunk_with_reasoning() -> None:
    """Test handling reasoning content."""
    delta_mock = Mock()
    delta_mock.content = "The answer is 42"
    delta_mock.role = "assistant"
    delta_mock.reasoning = "Let me think about this..."
    delta_mock.tool_calls = None

    choice_mock = Mock()
    choice_mock.delta = delta_mock
    choice_mock.index = 0
    choice_mock.finish_reason = "stop"

    together_chunk = Mock(spec=TogetherChatCompletionChunk)
    together_chunk.choices = [choice_mock]
    together_chunk.id = "test-id"
    together_chunk.created = int(datetime.now().timestamp())
    together_chunk.model = "test-model"
    together_chunk.usage = None

    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert result.choices[0].delta.content == "The answer is 42"
    assert result.choices[0].delta.reasoning is not None
    assert result.choices[0].delta.reasoning.content == "Let me think about this..."


def test_create_openai_chunk_with_usage() -> None:
    """Test handling usage metadata."""
    delta_mock = Mock()
    delta_mock.content = "Hello"
    delta_mock.role = "assistant"
    delta_mock.reasoning = None
    delta_mock.tool_calls = None

    choice_mock = Mock()
    choice_mock.delta = delta_mock
    choice_mock.index = 0
    choice_mock.finish_reason = "stop"

    usage_mock = Mock()
    usage_mock.prompt_tokens = 10
    usage_mock.completion_tokens = 5
    usage_mock.total_tokens = 15

    together_chunk = Mock(spec=TogetherChatCompletionChunk)
    together_chunk.choices = [choice_mock]
    together_chunk.id = "test-id"
    together_chunk.created = int(datetime.now().timestamp())
    together_chunk.model = "test-model"
    together_chunk.usage = usage_mock

    result = _create_openai_chunk_from_together_chunk(together_chunk)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15
