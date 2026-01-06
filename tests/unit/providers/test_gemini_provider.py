import base64
from contextlib import contextmanager
from typing import Any, Literal
from unittest.mock import AsyncMock, Mock, patch

import pytest
from google.genai import types

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.gemini import GeminiProvider
from any_llm.providers.gemini.base import REASONING_EFFORT_TO_THINKING_BUDGETS
from any_llm.providers.gemini.utils import _convert_messages, _convert_response_to_response_dict, _convert_tool_spec
from any_llm.types.completion import CompletionParams


@contextmanager
def mock_gemini_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.gemini.gemini.genai.Client") as mock_genai,
        patch("any_llm.providers.gemini.base._convert_response_to_response_dict") as mock_convert_response,
        patch.dict("os.environ", {"GOOGLE_PROJECT_ID": "test-project", "GOOGLE_REGION": "us-central1"}),
    ):
        mock_convert_response.return_value = {
            "id": "google_genai_response",
            "model": "gemini/genai",
            "created": 0,
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok", "tool_calls": None},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        # Set up the async method properly
        mock_client = mock_genai.return_value
        mock_client.aio.models.generate_content = AsyncMock()

        yield mock_genai


@pytest.mark.parametrize("env_var", ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
def test_gemini_initialization_with_env_var_api_key(env_var: str) -> None:
    """Test that the provider initializes correctly with API key from environment variable."""
    with patch.dict("os.environ", {env_var: "env-api-key"}, clear=True):
        provider = GeminiProvider()
        assert provider.client._api_client.api_key == "env-api-key"


@pytest.mark.asyncio
async def test_completion_with_system_instruction() -> None:
    """Test that completion works correctly with system_instruction."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"}]

    with mock_gemini_provider() as mock_genai:
        provider = GeminiProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        generation_config = call_kwargs["config"]
        contents = call_kwargs["contents"]

        assert len(contents) == 1
        assert generation_config.system_instruction == "You are a helpful assistant"


@pytest.mark.asyncio
async def test_completion_with_content_list() -> None:
    """Test that completion works correctly with content in list format."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

    with mock_gemini_provider() as mock_genai:
        provider = GeminiProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        contents = call_kwargs["contents"]

        assert contents[0].parts[0].text == "Hello"


@pytest.mark.parametrize(
    ("tool_choice", "expected_mode"),
    [
        ("auto", "AUTO"),
        ("required", "ANY"),
    ],
)
@pytest.mark.asyncio
async def test_completion_with_tool_choice_auto(tool_choice: str, expected_mode: str) -> None:
    """Test that completion correctly processes tool_choice='auto'."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_gemini_provider() as mock_genai:
        provider = GeminiProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages, tool_choice=tool_choice))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.tool_config.function_calling_config.mode.value == expected_mode


@pytest.mark.asyncio
async def test_completion_without_tool_choice() -> None:
    """Test that completion works correctly without tool_choice."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_gemini_provider() as mock_genai:
        provider = GeminiProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.tool_config is None


@pytest.mark.asyncio
async def test_completion_with_stream_and_response_format_raises() -> None:
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_gemini_provider():
        provider = GeminiProvider(api_key=api_key)
        with pytest.raises(UnsupportedParameterError):
            await provider._acompletion(
                CompletionParams(
                    model_id=model,
                    messages=messages,
                    stream=True,
                    response_format={"type": "json_object"},
                )
            )


@pytest.mark.asyncio
async def test_completion_with_parallel_tool_calls_raises() -> None:
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_gemini_provider():
        provider = GeminiProvider(api_key=api_key)
        with pytest.raises(UnsupportedParameterError):
            await provider._acompletion(
                CompletionParams(
                    model_id=model,
                    messages=messages,
                    parallel_tool_calls=True,
                )
            )


@pytest.mark.asyncio
async def test_completion_inside_agent_loop(agent_loop_messages: list[dict[str, Any]]) -> None:
    api_key = "test-api-key"
    model = "gemini-pro"

    with mock_gemini_provider() as mock_genai:
        provider = GeminiProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=agent_loop_messages))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args

        contents = call_kwargs["contents"]
        assert len(contents) == 3
        assert contents[0].role == "user"
        assert contents[1].role == "model"
        assert contents[2].role == "function"


@pytest.mark.parametrize(
    "reasoning_effort",
    [
        None,
        "low",
        "medium",
        "high",
    ],
)
@pytest.mark.asyncio
async def test_completion_with_custom_reasoning_effort(
    reasoning_effort: Literal["low", "medium", "high"] | None,
) -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_gemini_provider() as mock_genai:
        provider = GeminiProvider(api_key=api_key)
        await provider._acompletion(
            CompletionParams(model_id=model, messages=messages, reasoning_effort=reasoning_effort)
        )

        if reasoning_effort is None:
            expected_thinking = types.ThinkingConfig(include_thoughts=False)
        else:
            expected_thinking = types.ThinkingConfig(
                include_thoughts=True, thinking_budget=REASONING_EFFORT_TO_THINKING_BUDGETS[reasoning_effort]
            )
        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        assert call_kwargs["config"].thinking_config == expected_thinking


@pytest.mark.asyncio
async def test_completion_with_max_tokens_conversion() -> None:
    """Test that max_tokens parameter gets converted to max_output_tokens."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]
    max_tokens = 100

    with mock_gemini_provider() as mock_genai:
        provider = GeminiProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id=model, messages=messages, max_tokens=max_tokens))

        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        generation_config = call_kwargs["config"]

        assert generation_config.max_output_tokens == max_tokens


def test_convert_response_single_tool_call() -> None:
    """Test conversion of Google response with a single tool call to OpenAI format."""
    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content = Mock()
    mock_response.candidates[0].content.parts = [Mock()]

    mock_function_call = Mock()
    mock_function_call.name = "search_web"
    mock_function_call.args = {"query": "test query", "limit": 5}

    mock_response.candidates[0].content.parts[0].function_call = mock_function_call
    mock_response.candidates[0].content.parts[0].thought = None
    mock_response.candidates[0].content.parts[0].text = None

    mock_response.usage_metadata = Mock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 15
    mock_response.usage_metadata.total_token_count = 25

    response_dict = _convert_response_to_response_dict(mock_response)

    assert len(response_dict["choices"]) == 1
    choice = response_dict["choices"][0]

    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] is None
    assert choice["finish_reason"] == "tool_calls"
    assert choice["index"] == 0

    tool_calls = choice["message"]["tool_calls"]
    assert len(tool_calls) == 1

    tool_call = tool_calls[0]
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "search_web"
    assert tool_call["function"]["arguments"] == '{"query": "test query", "limit": 5}'
    assert tool_call["id"].startswith("call_")
    assert tool_call["id"].endswith("_0")


def test_convert_response_multiple_parallel_tool_calls() -> None:
    """Test conversion of Google response with multiple parallel tool calls to OpenAI format."""
    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content = Mock()

    mock_function_call_1 = Mock()
    mock_function_call_1.name = "search_web"
    mock_function_call_1.args = {"query": "test query"}

    mock_function_call_2 = Mock()
    mock_function_call_2.name = "get_weather"
    mock_function_call_2.args = {"location": "New York"}

    mock_function_call_3 = Mock()
    mock_function_call_3.name = "calculate"
    mock_function_call_3.args = {"expression": "2+2"}

    mock_part_1 = Mock()
    mock_part_1.function_call = mock_function_call_1
    mock_part_1.thought = None
    mock_part_1.text = None

    mock_part_2 = Mock()
    mock_part_2.function_call = mock_function_call_2
    mock_part_2.thought = None
    mock_part_2.text = None

    mock_part_3 = Mock()
    mock_part_3.function_call = mock_function_call_3
    mock_part_3.thought = None
    mock_part_3.text = None

    mock_response.candidates[0].content.parts = [mock_part_1, mock_part_2, mock_part_3]

    mock_response.usage_metadata = Mock()
    mock_response.usage_metadata.prompt_token_count = 20
    mock_response.usage_metadata.candidates_token_count = 30
    mock_response.usage_metadata.total_token_count = 50

    response_dict = _convert_response_to_response_dict(mock_response)

    assert len(response_dict["choices"]) == 1
    choice = response_dict["choices"][0]

    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] is None
    assert choice["finish_reason"] == "tool_calls"
    assert choice["index"] == 0

    tool_calls = choice["message"]["tool_calls"]
    assert len(tool_calls) == 3

    assert tool_calls[0]["function"]["name"] == "search_web"
    assert tool_calls[0]["function"]["arguments"] == '{"query": "test query"}'
    assert tool_calls[0]["id"].endswith("_0")

    assert tool_calls[1]["function"]["name"] == "get_weather"
    assert tool_calls[1]["function"]["arguments"] == '{"location": "New York"}'
    assert tool_calls[1]["id"].endswith("_1")

    assert tool_calls[2]["function"]["name"] == "calculate"
    assert tool_calls[2]["function"]["arguments"] == '{"expression": "2+2"}'
    assert tool_calls[2]["id"].endswith("_2")

    tool_call_ids = [tc["id"] for tc in tool_calls]
    assert len(set(tool_call_ids)) == 3


def test_convert_tool_spec_basic_mapping() -> None:
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search things",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query"},
                        # Array without items → should default items to {"type": "string"}
                        "opts": {"type": "array"},
                        # Array with items → should be preserved
                        "count_list": {"type": "array", "items": {"type": "integer"}},
                        # Enum should be preserved
                        "mode": {"type": "string", "enum": ["a", "b"]},
                        # additionalProperties should be dropped
                        "config": {"type": "object", "additionalProperties": {"type": "integer"}},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    tools = _convert_tool_spec(openai_tools)

    assert len(tools) == 1
    assert tools[0].function_declarations[0].name == "search"  # type: ignore[index]
    assert tools[0].function_declarations[0].description == "Search things"  # type: ignore[index]
    assert tools[0].function_declarations[0].parameters.type == "OBJECT"  # type: ignore[index, union-attr]
    assert tools[0].function_declarations[0].parameters.properties["query"].type == "STRING"  # type: ignore[union-attr, index]
    assert tools[0].function_declarations[0].parameters.properties["opts"].type == "ARRAY"  # type: ignore[union-attr, index]
    assert tools[0].function_declarations[0].parameters.properties["count_list"].type == "ARRAY"  # type: ignore[union-attr, index]
    assert tools[0].function_declarations[0].parameters.properties["mode"].type == "STRING"  # type: ignore[union-attr, index]
    assert tools[0].function_declarations[0].parameters.properties["config"].type == "OBJECT"  # type: ignore[union-attr, index]
    assert "additionalProperties" not in tools[0].function_declarations[0].parameters.properties["config"]  # type: ignore[union-attr, index]


@pytest.mark.asyncio
async def test_gemini_with_built_in_tools() -> None:
    """Test that built-in tools are added correctly when specified."""
    api_key = "test-api-key"
    model = "gemini-pro"
    messages = [{"role": "user", "content": "Hello"}]
    google_search = types.Tool(google_search=types.GoogleSearch())
    with mock_gemini_provider() as mock_genai:
        provider = GeminiProvider(api_key=api_key)
        await provider.acompletion(model=model, messages=messages, tools=[google_search])  # type: ignore[arg-type]
        _, call_kwargs = mock_genai.return_value.aio.models.generate_content.call_args
        generation_config = call_kwargs["config"]
        assert generation_config.tools is not None
        assert len(generation_config.tools) == 1
        assert generation_config.tools[0] == google_search


@pytest.mark.asyncio
async def test_streaming_completion_includes_usage_data() -> None:
    """Test that streaming chunks include usage metadata when available."""
    from any_llm.providers.gemini.utils import _create_openai_chunk_from_google_chunk

    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content = Mock()
    mock_response.candidates[0].content.parts = [Mock()]
    mock_response.candidates[0].content.parts[0].text = "Hello"
    mock_response.candidates[0].content.parts[0].thought = None
    mock_response.candidates[0].content.parts[0].function_call = None
    mock_response.candidates[0].finish_reason = Mock()
    mock_response.candidates[0].finish_reason.value = "STOP"
    mock_response.model_version = "gemini-2.5-flash"

    mock_response.usage_metadata = Mock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.usage_metadata.total_token_count = 15

    chunk = _create_openai_chunk_from_google_chunk(mock_response)

    assert chunk.usage is not None, "Usage data should be included in streaming chunks"
    assert chunk.usage.prompt_tokens == 10
    assert chunk.usage.completion_tokens == 5
    assert chunk.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_streaming_completion_without_usage_metadata() -> None:
    """Test that streaming chunks handle missing usage metadata gracefully."""
    from any_llm.providers.gemini.utils import _create_openai_chunk_from_google_chunk

    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content = Mock()
    mock_response.candidates[0].content.parts = [Mock()]
    mock_response.candidates[0].content.parts[0].text = "Hello"
    mock_response.candidates[0].content.parts[0].thought = None
    mock_response.candidates[0].content.parts[0].function_call = None
    mock_response.candidates[0].finish_reason = Mock()
    mock_response.candidates[0].finish_reason.value = None
    mock_response.model_version = "gemini-2.5-flash"

    mock_response.usage_metadata = None

    chunk = _create_openai_chunk_from_google_chunk(mock_response)

    assert chunk.usage is None, "Usage should be None when metadata is not available"
    assert chunk.choices[0].delta.content == "Hello"


def test_streaming_completion_with_tool_call() -> None:
    """Test that streaming chunks properly handle function calls."""
    from any_llm.providers.gemini.utils import _create_openai_chunk_from_google_chunk

    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content = Mock()

    mock_function_call = Mock()
    mock_function_call.name = "get_weather"
    mock_function_call.args = {"location": "Paris"}

    mock_part = Mock()
    mock_part.function_call = mock_function_call
    mock_part.thought = None
    mock_part.text = None

    mock_response.candidates[0].content.parts = [mock_part]
    mock_response.candidates[0].finish_reason = Mock()
    mock_response.candidates[0].finish_reason.value = "STOP"
    mock_response.model_version = "gemini-2.5-flash"
    mock_response.usage_metadata = None

    chunk = _create_openai_chunk_from_google_chunk(mock_response)

    assert chunk.choices[0].delta.tool_calls is not None, "Tool calls should be present in streaming chunk"
    assert len(chunk.choices[0].delta.tool_calls) == 1
    tool_call = chunk.choices[0].delta.tool_calls[0]
    assert tool_call.function is not None
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"location": "Paris"}'


def test_streaming_completion_with_multiple_tool_calls() -> None:
    """Test that streaming chunks handle multiple parallel tool calls."""
    from any_llm.providers.gemini.utils import _create_openai_chunk_from_google_chunk

    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content = Mock()

    mock_function_call_1 = Mock()
    mock_function_call_1.name = "get_weather"
    mock_function_call_1.args = {"location": "Paris"}

    mock_function_call_2 = Mock()
    mock_function_call_2.name = "get_weather"
    mock_function_call_2.args = {"location": "London"}

    mock_part_1 = Mock()
    mock_part_1.function_call = mock_function_call_1
    mock_part_1.thought = None
    mock_part_1.text = None

    mock_part_2 = Mock()
    mock_part_2.function_call = mock_function_call_2
    mock_part_2.thought = None
    mock_part_2.text = None

    mock_response.candidates[0].content.parts = [mock_part_1, mock_part_2]
    mock_response.candidates[0].finish_reason = Mock()
    mock_response.candidates[0].finish_reason.value = "STOP"
    mock_response.model_version = "gemini-2.5-flash"
    mock_response.usage_metadata = None

    chunk = _create_openai_chunk_from_google_chunk(mock_response)

    assert chunk.choices[0].delta.tool_calls is not None, "Tool calls should be present"
    assert len(chunk.choices[0].delta.tool_calls) == 2
    assert chunk.choices[0].delta.tool_calls[0].function is not None
    assert chunk.choices[0].delta.tool_calls[0].function.name == "get_weather"
    assert chunk.choices[0].delta.tool_calls[1].function is not None
    assert chunk.choices[0].delta.tool_calls[1].function.name == "get_weather"


def test_convert_response_preserves_thought_signature() -> None:
    import base64

    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content = Mock()

    mock_function_call = Mock()
    mock_function_call.name = "get_weather"
    mock_function_call.args = {"location": "Paris"}

    mock_part = Mock()
    mock_part.function_call = mock_function_call
    mock_part.thought = None
    mock_part.text = None
    # Gemini returns thought_signature as bytes
    mock_part.thought_signature = b"test-signature-bytes"

    mock_response.candidates[0].content.parts = [mock_part]
    mock_response.usage_metadata = Mock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 15
    mock_response.usage_metadata.total_token_count = 25

    response_dict = _convert_response_to_response_dict(mock_response)

    tool_calls = response_dict["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert "extra_content" in tool_calls[0]
    assert tool_calls[0]["extra_content"]["google"]["thought_signature"] == base64.b64encode(
        b"test-signature-bytes"
    ).decode("utf-8")


def test_convert_response_no_thought_signature() -> None:
    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content = Mock()

    mock_function_call = Mock()
    mock_function_call.name = "get_weather"
    mock_function_call.args = {"location": "Paris"}

    mock_part = Mock()
    mock_part.function_call = mock_function_call
    mock_part.thought = None
    mock_part.text = None
    mock_part.thought_signature = None

    mock_response.candidates[0].content.parts = [mock_part]
    mock_response.usage_metadata = Mock()
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 15
    mock_response.usage_metadata.total_token_count = 25

    response_dict = _convert_response_to_response_dict(mock_response)

    tool_calls = response_dict["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert "extra_content" not in tool_calls[0] or tool_calls[0].get("extra_content") is None


def test_convert_messages_with_thought_signature_in_extra_content() -> None:
    # Use valid base64 that round-trips correctly
    original_bytes = b"test-signature-bytes"
    base64_signature = base64.b64encode(original_bytes).decode("utf-8")

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What is the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                    "extra_content": {"google": {"thought_signature": base64_signature}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "name": "get_weather", "content": '{"temp": "20C"}'},
    ]

    formatted_messages, _ = _convert_messages(messages)

    # The assistant message with tool_calls should have the thought_signature
    # SDK decodes base64 string to bytes
    assistant_message = formatted_messages[1]
    assert assistant_message.role == "model"
    assert assistant_message.parts is not None
    assert len(assistant_message.parts) == 1
    assert assistant_message.parts[0].thought_signature == original_bytes


def test_convert_messages_without_thought_signature_uses_skip_sentinel() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "What is the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_123", "name": "get_weather", "content": '{"temp": "20C"}'},
    ]

    formatted_messages, _ = _convert_messages(messages)

    # The assistant message should use skip sentinel for first tool call
    # SDK may decode the sentinel string, so just verify it's set (not None)
    assistant_message = formatted_messages[1]
    assert assistant_message.role == "model"
    assert assistant_message.parts is not None
    assert len(assistant_message.parts) == 1
    assert assistant_message.parts[0].thought_signature is not None


def test_convert_messages_parallel_tool_calls_only_first_gets_skip_sentinel() -> None:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Get weather for Paris and London"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "London"}'},
                },
            ],
        },
    ]

    formatted_messages, _ = _convert_messages(messages)

    assistant_message = formatted_messages[1]
    assert assistant_message.parts is not None
    assert len(assistant_message.parts) == 2
    # First tool call should have skip sentinel set (SDK decodes to bytes)
    assert assistant_message.parts[0].thought_signature is not None
    # Second tool call should have None (no sentinel)
    assert assistant_message.parts[1].thought_signature is None
