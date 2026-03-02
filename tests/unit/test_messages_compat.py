"""Tests for bidirectional Anthropic Messages ↔ OpenAI Chat Completions conversion."""

import json

import pytest

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
    Function,
    Reasoning,
)
from any_llm.types.messages import MessageContentBlock, MessageResponse, MessagesParams, MessageUsage
from any_llm.utils.messages_compat import (
    StreamingState,
    chat_completion_chunk_to_message_stream_events,
    chat_completion_to_message_response,
    messages_params_to_completion_params,
)


def test_basic_text_message_conversion() -> None:
    """Test converting a simple text message from Anthropic to OpenAI format."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
    )
    result = messages_params_to_completion_params(params)
    assert result["model_id"] == "claude-3-5-sonnet"
    assert result["max_tokens"] == 1024
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == "Hello"


def test_system_message_prepended() -> None:
    """Test that system message is prepended as a system role message."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=1024,
        system="You are helpful.",
    )
    result = messages_params_to_completion_params(params)
    assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
    assert result["messages"][1]["role"] == "user"


def test_tool_conversion_to_openai() -> None:
    """Test Anthropic tool format → OpenAI function tool format."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "What's the weather?"}],
        max_tokens=1024,
        tools=[{
            "name": "get_weather",
            "description": "Get weather info",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        }],
    )
    result = messages_params_to_completion_params(params)
    assert len(result["tools"]) == 1
    tool = result["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    assert tool["function"]["parameters"]["type"] == "object"


def test_tool_choice_auto() -> None:
    """Test tool_choice auto conversion."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1024,
        tool_choice={"type": "auto"},
    )
    result = messages_params_to_completion_params(params)
    assert result["tool_choice"] == "auto"


def test_tool_choice_any() -> None:
    """Test tool_choice 'any' → 'required' conversion."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1024,
        tool_choice={"type": "any"},
    )
    result = messages_params_to_completion_params(params)
    assert result["tool_choice"] == "required"


def test_tool_choice_none() -> None:
    """Test tool_choice 'none' conversion."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1024,
        tool_choice={"type": "none"},
    )
    result = messages_params_to_completion_params(params)
    assert result["tool_choice"] == "none"


def test_tool_choice_specific_tool() -> None:
    """Test tool_choice for a specific tool → OpenAI function format."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1024,
        tool_choice={"type": "tool", "name": "get_weather"},
    )
    result = messages_params_to_completion_params(params)
    assert result["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}


def test_stop_sequences_to_stop() -> None:
    """Test stop_sequences → stop conversion."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1024,
        stop_sequences=["END", "STOP"],
    )
    result = messages_params_to_completion_params(params)
    assert result["stop"] == ["END", "STOP"]


def test_thinking_enabled_to_reasoning_effort() -> None:
    """Test thinking config → reasoning_effort conversion."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1024,
        thinking={"type": "enabled", "budget_tokens": 8192},
    )
    result = messages_params_to_completion_params(params)
    assert result["reasoning_effort"] == "medium"


def test_thinking_disabled_to_reasoning_none() -> None:
    """Test thinking disabled → reasoning_effort none."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1024,
        thinking={"type": "disabled"},
    )
    result = messages_params_to_completion_params(params)
    assert result["reasoning_effort"] == "none"


def test_tool_use_message_conversion() -> None:
    """Test assistant tool_use content blocks → OpenAI tool_calls format."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "call_123", "name": "get_weather", "input": {"city": "London"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_123", "content": "Sunny, 20°C"},
                ],
            },
        ],
        max_tokens=1024,
    )
    result = messages_params_to_completion_params(params)
    # First message: user
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][0]["content"] == "What's the weather?"

    # Assistant message should have tool_calls
    assistant_msg = result["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert len(assistant_msg["tool_calls"]) == 1
    assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(assistant_msg["tool_calls"][0]["function"]["arguments"]) == {"city": "London"}

    # Tool result → role: tool
    tool_msg = result["messages"][2]
    assert tool_msg["role"] == "tool"
    assert tool_msg["tool_call_id"] == "call_123"
    assert tool_msg["content"] == "Sunny, 20°C"


def test_image_block_conversion() -> None:
    """Test image content blocks → OpenAI image_url format."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc123"}},
            ],
        }],
        max_tokens=1024,
    )
    result = messages_params_to_completion_params(params)
    user_msg = result["messages"][0]
    assert user_msg["role"] == "user"
    assert len(user_msg["content"]) == 2
    assert user_msg["content"][0]["type"] == "text"
    assert user_msg["content"][1]["type"] == "image_url"
    assert user_msg["content"][1]["image_url"]["url"] == "data:image/png;base64,abc123"


def test_chat_completion_text_response_to_message() -> None:
    """Test converting a text ChatCompletion to MessageResponse."""
    completion = ChatCompletion(
        id="chatcmpl-123",
        model="gpt-4",
        created=0,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="Hello!"),
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    result = chat_completion_to_message_response(completion)
    assert result.id == "chatcmpl-123"
    assert result.role == "assistant"
    assert result.stop_reason == "end_turn"
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == "Hello!"
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5


def test_chat_completion_tool_calls_response_to_message() -> None:
    """Test converting a tool_calls ChatCompletion to MessageResponse with tool_use blocks."""
    completion = ChatCompletion(
        id="chatcmpl-456",
        model="gpt-4",
        created=0,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id="call_abc",
                            type="function",
                            function=Function(name="get_weather", arguments='{"city": "London"}'),
                        )
                    ],
                ),
            )
        ],
        usage=CompletionUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
    )
    result = chat_completion_to_message_response(completion)
    assert result.stop_reason == "tool_use"
    assert len(result.content) == 1
    assert result.content[0].type == "tool_use"
    assert result.content[0].name == "get_weather"
    assert result.content[0].input == {"city": "London"}
    assert result.content[0].id == "call_abc"


def test_chat_completion_reasoning_response_to_message() -> None:
    """Test converting a ChatCompletion with reasoning to MessageResponse with thinking block."""
    completion = ChatCompletion(
        id="chatcmpl-789",
        model="o1",
        created=0,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    role="assistant",
                    content="The answer is 42.",
                    reasoning=Reasoning(content="Let me think about this..."),
                ),
            )
        ],
        usage=CompletionUsage(prompt_tokens=15, completion_tokens=20, total_tokens=35),
    )
    result = chat_completion_to_message_response(completion)
    assert len(result.content) == 2
    assert result.content[0].type == "thinking"
    assert result.content[0].thinking == "Let me think about this..."
    assert result.content[1].type == "text"
    assert result.content[1].text == "The answer is 42."


def test_finish_reason_mapping() -> None:
    """Test all finish_reason → stop_reason mappings."""
    for finish, expected_stop in [("stop", "end_turn"), ("length", "max_tokens"), ("tool_calls", "tool_use")]:
        completion = ChatCompletion(
            id="test",
            model="test",
            created=0,
            object="chat.completion",
            choices=[Choice(index=0, finish_reason=finish, message=ChatCompletionMessage(role="assistant", content=""))],
        )
        result = chat_completion_to_message_response(completion)
        assert result.stop_reason == expected_stop, f"finish_reason={finish} should map to stop_reason={expected_stop}"


def test_streaming_text_events() -> None:
    """Test streaming conversion produces correct event lifecycle for text."""
    state = StreamingState()

    # First chunk with content
    chunk1 = ChatCompletionChunk(
        id="chunk-1",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(content="Hello"), finish_reason=None)],
    )
    events1 = chat_completion_chunk_to_message_stream_events(chunk1, state)
    types1 = [e.type for e in events1]
    assert "message_start" in types1
    assert "content_block_start" in types1
    assert "content_block_delta" in types1

    # Verify text delta content
    text_delta = next(e for e in events1 if e.type == "content_block_delta")
    assert text_delta.delta is not None
    assert text_delta.delta["text"] == "Hello"

    # Final chunk with finish_reason
    chunk2 = ChatCompletionChunk(
        id="chunk-2",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="stop")],
    )
    events2 = chat_completion_chunk_to_message_stream_events(chunk2, state)
    types2 = [e.type for e in events2]
    assert "content_block_stop" in types2
    assert "message_delta" in types2
    assert "message_stop" in types2


def test_streaming_tool_call_events() -> None:
    """Test streaming conversion for tool calls."""
    from any_llm.types.completion import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction

    state = StreamingState()

    # First chunk: message start + tool call start
    chunk1 = ChatCompletionChunk(
        id="chunk-1",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            id="call_123",
                            function=ChoiceDeltaToolCallFunction(name="get_weather", arguments=""),
                        )
                    ]
                ),
                finish_reason=None,
            )
        ],
    )
    events1 = chat_completion_chunk_to_message_stream_events(chunk1, state)
    assert any(e.type == "content_block_start" for e in events1)
    start_event = next(e for e in events1 if e.type == "content_block_start")
    assert start_event.content_block is not None
    assert start_event.content_block.type == "tool_use"
    assert start_event.content_block.name == "get_weather"

    # Second chunk: tool call arguments
    chunk2 = ChatCompletionChunk(
        id="chunk-2",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(
                    tool_calls=[
                        ChoiceDeltaToolCall(
                            index=0,
                            function=ChoiceDeltaToolCallFunction(arguments='{"city":"London"}'),
                        )
                    ]
                ),
                finish_reason=None,
            )
        ],
    )
    events2 = chat_completion_chunk_to_message_stream_events(chunk2, state)
    delta_event = next(e for e in events2 if e.type == "content_block_delta")
    assert delta_event.delta is not None
    assert delta_event.delta["partial_json"] == '{"city":"London"}'


def test_optional_params_not_included_when_none() -> None:
    """Test that optional params like temperature aren't included when not set."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1024,
    )
    result = messages_params_to_completion_params(params)
    assert "temperature" not in result
    assert "top_p" not in result
    assert "stop" not in result
    assert "tools" not in result


def test_temperature_and_top_p_passed_through() -> None:
    """Test that temperature and top_p are passed when set."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1024,
        temperature=0.7,
        top_p=0.9,
    )
    result = messages_params_to_completion_params(params)
    assert result["temperature"] == 0.7
    assert result["top_p"] == 0.9
