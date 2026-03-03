"""Tests for bidirectional Anthropic Messages ↔ OpenAI Chat Completions conversion."""

import json
from typing import Any

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
from any_llm.types.messages import MessagesParams
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
        tools=[
            {
                "name": "get_weather",
                "description": "Get weather info",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            }
        ],
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
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc123"}},
                ],
            }
        ],
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
    from any_llm.utils.messages_compat import _finish_reason_to_stop_reason

    assert _finish_reason_to_stop_reason("stop") == "end_turn"
    assert _finish_reason_to_stop_reason("length") == "max_tokens"
    assert _finish_reason_to_stop_reason("tool_calls") == "tool_use"


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


def test_budget_to_reasoning_effort_minimal() -> None:
    """Test budget <= 1024 maps to 'minimal'."""
    from any_llm.utils.messages_compat import _budget_to_reasoning_effort

    assert _budget_to_reasoning_effort(512) == "minimal"
    assert _budget_to_reasoning_effort(1024) == "minimal"


def test_budget_to_reasoning_effort_low() -> None:
    """Test budget 1025-2048 maps to 'low'."""
    from any_llm.utils.messages_compat import _budget_to_reasoning_effort

    assert _budget_to_reasoning_effort(1025) == "low"
    assert _budget_to_reasoning_effort(2048) == "low"


def test_budget_to_reasoning_effort_high() -> None:
    """Test budget 8193-24576 maps to 'high'."""
    from any_llm.utils.messages_compat import _budget_to_reasoning_effort

    assert _budget_to_reasoning_effort(8193) == "high"
    assert _budget_to_reasoning_effort(24576) == "high"


def test_budget_to_reasoning_effort_xhigh() -> None:
    """Test budget > 24576 maps to 'xhigh'."""
    from any_llm.utils.messages_compat import _budget_to_reasoning_effort

    assert _budget_to_reasoning_effort(24577) == "xhigh"
    assert _budget_to_reasoning_effort(100000) == "xhigh"


def test_tool_choice_unknown_type_defaults_to_auto() -> None:
    """Test unknown tool_choice type falls back to 'auto'."""
    from any_llm.utils.messages_compat import _convert_tool_choice_to_openai

    assert _convert_tool_choice_to_openai({"type": "unknown_type"}) == "auto"


def test_finish_reason_none_maps_to_end_turn() -> None:
    """Test None finish_reason maps to 'end_turn'."""
    from any_llm.utils.messages_compat import _finish_reason_to_stop_reason

    assert _finish_reason_to_stop_reason(None) == "end_turn"


def test_finish_reason_content_filter() -> None:
    """Test content_filter and function_call finish_reason mappings."""
    from any_llm.utils.messages_compat import _finish_reason_to_stop_reason

    assert _finish_reason_to_stop_reason("content_filter") == "end_turn"
    assert _finish_reason_to_stop_reason("function_call") == "tool_use"


def test_finish_reason_unknown_defaults_to_end_turn() -> None:
    """Test unknown finish_reason defaults to 'end_turn'."""
    from any_llm.utils.messages_compat import _finish_reason_to_stop_reason

    assert _finish_reason_to_stop_reason("some_unknown_reason") == "end_turn"


def test_message_non_list_non_string_content() -> None:
    """Test conversion when content is neither string nor list (e.g., None)."""
    from any_llm.utils.messages_compat import _convert_message_to_openai

    result = _convert_message_to_openai({"role": "user", "content": None})
    assert result == [{"role": "user", "content": None}]


def test_message_unknown_role_with_blocks() -> None:
    """Test conversion of unknown role with list content passes through as-is."""
    from any_llm.utils.messages_compat import _convert_message_to_openai

    blocks = [{"type": "text", "text": "hi"}]
    result = _convert_message_to_openai({"role": "developer", "content": blocks})
    assert result == [{"role": "developer", "content": blocks}]


def test_assistant_blocks_text_only() -> None:
    """Test assistant message with only text blocks and no tool_calls."""
    from any_llm.utils.messages_compat import _convert_assistant_blocks_to_openai

    result = _convert_assistant_blocks_to_openai(
        [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ]
    )
    assert len(result) == 1
    assert result[0]["content"] == "Hello world"
    assert "tool_calls" not in result[0]


def test_assistant_blocks_tool_only() -> None:
    """Test assistant message with only tool_use blocks has content=None."""
    from any_llm.utils.messages_compat import _convert_assistant_blocks_to_openai

    result = _convert_assistant_blocks_to_openai(
        [
            {"type": "tool_use", "id": "call_1", "name": "fn", "input": {}},
        ]
    )
    assert result[0]["content"] is None
    assert len(result[0]["tool_calls"]) == 1


def test_assistant_blocks_mixed_text_and_tool() -> None:
    """Test assistant message with both text and tool_use blocks."""
    from any_llm.utils.messages_compat import _convert_assistant_blocks_to_openai

    result = _convert_assistant_blocks_to_openai(
        [
            {"type": "text", "text": "Let me check"},
            {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "test"}},
        ]
    )
    assert result[0]["content"] == "Let me check"
    assert len(result[0]["tool_calls"]) == 1
    assert result[0]["tool_calls"][0]["function"]["name"] == "search"


def test_assistant_blocks_unknown_type_ignored() -> None:
    """Test unknown block types in assistant messages are silently ignored."""
    from any_llm.utils.messages_compat import _convert_assistant_blocks_to_openai

    result = _convert_assistant_blocks_to_openai(
        [
            {"type": "unknown_block", "data": "something"},
        ]
    )
    assert result[0]["content"] is None
    assert "tool_calls" not in result[0]


def test_user_blocks_tool_result_with_list_content() -> None:
    """Test tool_result block where content is a list of content blocks."""
    from any_llm.utils.messages_compat import _convert_user_blocks_to_openai

    result = _convert_user_blocks_to_openai(
        [
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": [
                    {"type": "text", "text": "Result: "},
                    {"type": "text", "text": "42"},
                    {"type": "image", "data": "ignored"},
                ],
            },
        ]
    )
    assert len(result) == 1
    assert result[0]["role"] == "tool"
    assert result[0]["content"] == "Result: 42"


def test_user_blocks_unknown_type_passed_through() -> None:
    """Test unknown block type in user message is kept as-is in content."""
    from any_llm.utils.messages_compat import _convert_user_blocks_to_openai

    result = _convert_user_blocks_to_openai(
        [
            {"type": "custom_block", "data": "value"},
        ]
    )
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == [{"type": "custom_block", "data": "value"}]


def test_user_blocks_mixed_text_then_tool_result() -> None:
    """Test user blocks with text followed by tool_result flushes text first."""
    from any_llm.utils.messages_compat import _convert_user_blocks_to_openai

    result = _convert_user_blocks_to_openai(
        [
            {"type": "text", "text": "Here's context"},
            {"type": "tool_result", "tool_use_id": "call_1", "content": "Done"},
        ]
    )
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"][0]["text"] == "Here's context"
    assert result[1]["role"] == "tool"
    assert result[1]["content"] == "Done"


def test_image_url_source_type() -> None:
    """Test image block with URL source type."""
    from any_llm.utils.messages_compat import _convert_user_blocks_to_openai

    result = _convert_user_blocks_to_openai(
        [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}},
        ]
    )
    assert result[0]["content"][0]["type"] == "image_url"
    assert result[0]["content"][0]["image_url"]["url"] == "https://example.com/img.png"


def test_chat_completion_no_choices() -> None:
    """Test converting a ChatCompletion with empty choices."""
    completion = ChatCompletion(
        id="test",
        model="test",
        created=0,
        object="chat.completion",
        choices=[],
    )
    result = chat_completion_to_message_response(completion)
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == ""
    assert result.stop_reason == "end_turn"
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


def test_chat_completion_no_usage() -> None:
    """Test converting a ChatCompletion with no usage field."""
    completion = ChatCompletion(
        id="test",
        model="test",
        created=0,
        object="chat.completion",
        choices=[Choice(index=0, finish_reason="stop", message=ChatCompletionMessage(role="assistant", content="Hi"))],
    )
    result = chat_completion_to_message_response(completion)
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


def test_chat_completion_tool_calls_invalid_json() -> None:
    """Test tool call with invalid JSON arguments falls back to empty dict."""
    completion = ChatCompletion(
        id="test",
        model="test",
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
                            id="call_1", type="function", function=Function(name="fn", arguments="not valid json{")
                        )
                    ],
                ),
            )
        ],
    )
    result = chat_completion_to_message_response(completion)
    assert result.content[0].input == {}


def test_chat_completion_tool_calls_empty_arguments() -> None:
    """Test tool call with empty/None arguments falls back to empty dict."""
    completion = ChatCompletion(
        id="test",
        model="test",
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
                            id="call_1", type="function", function=Function(name="fn", arguments="")
                        )
                    ],
                ),
            )
        ],
    )
    result = chat_completion_to_message_response(completion)
    assert result.content[0].input == {}


def test_streaming_usage_tracking() -> None:
    """Test that streaming state tracks usage from chunks."""
    state = StreamingState()

    chunk = ChatCompletionChunk(
        id="chunk-1",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(content="Hi"), finish_reason=None)],
        usage=CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )
    chat_completion_chunk_to_message_stream_events(chunk, state)
    assert state.input_tokens == 100
    assert state.output_tokens == 50


def test_streaming_no_choices_returns_early() -> None:
    """Test chunk with no choices only returns message_start if first."""
    state = StreamingState()

    chunk = ChatCompletionChunk(
        id="chunk-1",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[],
    )
    events = chat_completion_chunk_to_message_stream_events(chunk, state)
    assert len(events) == 1
    assert events[0].type == "message_start"


def test_streaming_reasoning_then_text_transition() -> None:
    """Test transition from reasoning block to text block emits block_stop + block_start."""
    state = StreamingState()

    chunk1 = ChatCompletionChunk(
        id="chunk-1",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(reasoning=Reasoning(content="thinking...")),
                finish_reason=None,
            )
        ],
    )
    events1 = chat_completion_chunk_to_message_stream_events(chunk1, state)
    assert state.current_block_type == "thinking"
    assert any(e.type == "content_block_start" for e in events1)
    thinking_start = next(e for e in events1 if e.type == "content_block_start")
    assert thinking_start.content_block is not None
    assert thinking_start.content_block.type == "thinking"

    chunk2 = ChatCompletionChunk(
        id="chunk-2",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(content="Answer"), finish_reason=None)],
    )
    events2 = chat_completion_chunk_to_message_stream_events(chunk2, state)
    types2 = [e.type for e in events2]
    assert "content_block_stop" in types2
    assert "content_block_start" in types2
    assert state.current_block_type == "text"


def test_streaming_empty_content_no_delta() -> None:
    """Test that empty string content emits block_start but not a delta."""
    state = StreamingState()

    chunk = ChatCompletionChunk(
        id="chunk-1",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(content=""), finish_reason=None)],
    )
    events = chat_completion_chunk_to_message_stream_events(chunk, state)
    types = [e.type for e in events]
    assert "content_block_start" in types
    assert "content_block_delta" not in types


def test_streaming_finish_reason_length() -> None:
    """Test streaming with finish_reason 'length' emits correct stop_reason."""
    state = StreamingState()
    state.started = True
    state.current_block_index = 0
    state.current_block_type = "text"

    chunk = ChatCompletionChunk(
        id="chunk-1",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(), finish_reason="length")],
    )
    events = chat_completion_chunk_to_message_stream_events(chunk, state)
    delta_event = next(e for e in events if e.type == "message_delta")
    assert delta_event.delta is not None
    assert delta_event.delta["stop_reason"] == "max_tokens"


def test_close_current_block_when_none() -> None:
    """Test _close_current_block does nothing when no block is open."""
    from any_llm.utils.messages_compat import _close_current_block

    state = StreamingState()
    events: list[Any] = []
    _close_current_block(state, events)
    assert len(events) == 0


def test_close_current_block_when_open() -> None:
    """Test _close_current_block emits stop event when block is open."""
    from any_llm.utils.messages_compat import _close_current_block

    state = StreamingState()
    state.current_block_type = "text"
    state.current_block_index = 2
    events: list[Any] = []
    _close_current_block(state, events)
    assert len(events) == 1
    assert events[0].type == "content_block_stop"
    assert events[0].index == 2
    assert state.current_block_type is None


def test_stream_param_passed_through() -> None:
    """Test that stream=True is included in conversion output."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1024,
        stream=True,
    )
    result = messages_params_to_completion_params(params)
    assert result["stream"] is True


def test_thinking_unknown_type_ignored() -> None:
    """Test that thinking config with unknown type produces no reasoning_effort."""
    params = MessagesParams(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=1024,
        thinking={"type": "something_else"},
    )
    result = messages_params_to_completion_params(params)
    assert "reasoning_effort" not in result


def test_tool_call_without_function_attribute_skipped() -> None:
    """Test that a custom tool call (no function attribute) is skipped."""
    from openai.types.chat.chat_completion_message_custom_tool_call import (
        ChatCompletionMessageCustomToolCall,
        Custom,
    )

    custom_tc = ChatCompletionMessageCustomToolCall(id="tc_1", type="custom", custom=Custom(name="my_tool", input="{}"))
    completion = ChatCompletion(
        id="test",
        model="test",
        created=0,
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[custom_tc],
                ),
            )
        ],
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
    )
    result = chat_completion_to_message_response(completion)
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == ""


def test_streaming_consecutive_text_deltas_no_extra_block_start() -> None:
    """Test that consecutive text deltas don't open a new block."""
    state = StreamingState()

    chunk1 = ChatCompletionChunk(
        id="c1",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(content="Hello"), finish_reason=None)],
    )
    events1 = chat_completion_chunk_to_message_stream_events(chunk1, state)
    assert sum(1 for e in events1 if e.type == "content_block_start") == 1

    chunk2 = ChatCompletionChunk(
        id="c2",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(content=" world"), finish_reason=None)],
    )
    events2 = chat_completion_chunk_to_message_stream_events(chunk2, state)
    assert not any(e.type == "content_block_start" for e in events2)
    assert any(e.type == "content_block_delta" for e in events2)


def test_streaming_consecutive_thinking_deltas_no_extra_block_start() -> None:
    """Test that consecutive thinking deltas don't open a new block."""
    state = StreamingState()

    chunk1 = ChatCompletionChunk(
        id="c1",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(reasoning=Reasoning(content="first thought")),
                finish_reason=None,
            )
        ],
    )
    events1 = chat_completion_chunk_to_message_stream_events(chunk1, state)
    assert sum(1 for e in events1 if e.type == "content_block_start") == 1

    chunk2 = ChatCompletionChunk(
        id="c2",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(reasoning=Reasoning(content="more thinking")),
                finish_reason=None,
            )
        ],
    )
    events2 = chat_completion_chunk_to_message_stream_events(chunk2, state)
    assert not any(e.type == "content_block_start" for e in events2)
    assert any(e.type == "content_block_delta" for e in events2)


def test_streaming_usage_with_zero_tokens() -> None:
    """Test that zero-value token counts don't overwrite previously tracked values."""
    state = StreamingState()
    state.started = True
    state.input_tokens = 100
    state.output_tokens = 50

    chunk = ChatCompletionChunk(
        id="c1",
        model="gpt-4",
        created=0,
        object="chat.completion.chunk",
        choices=[ChunkChoice(index=0, delta=ChoiceDelta(content="Hi"), finish_reason=None)],
        usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
    chat_completion_chunk_to_message_stream_events(chunk, state)
    assert state.input_tokens == 100
    assert state.output_tokens == 50
