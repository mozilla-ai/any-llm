from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm.providers.huggingface.huggingface import HuggingfaceProvider
from any_llm.providers.huggingface.utils import _create_openai_chunk_from_huggingface_chunk
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams
from any_llm.types.responses import Response, ResponsesParams


@contextmanager
def mock_huggingface_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.huggingface.huggingface.AsyncInferenceClient") as mock_huggingface,
    ):
        async_mock = AsyncMock()
        mock_huggingface.return_value = async_mock
        async_mock.chat_completion.return_value = {
            "id": "hf-response-id",
            "created": 0,
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok", "tool_calls": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        yield mock_huggingface


@pytest.mark.asyncio
async def test_huggingface_with_api_base() -> None:
    api_key = "test-api-key"
    api_base = "https://test.huggingface.co"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_huggingface_provider() as mock_huggingface:
        provider = HuggingfaceProvider(api_key=api_key, api_base=api_base)
        await provider._acompletion(CompletionParams(model_id="model-id", messages=messages, max_tokens=100))
        mock_huggingface.assert_called_with(base_url=api_base, token=api_key)


@pytest.mark.asyncio
async def test_huggingface_with_max_tokens() -> None:
    api_key = "test-api-key"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_huggingface_provider() as mock_huggingface:
        provider = HuggingfaceProvider(api_key=api_key)
        await provider._acompletion(CompletionParams(model_id="model-id", messages=messages, max_tokens=100))

        mock_huggingface.assert_called_with(base_url=None, token=api_key)


@pytest.mark.asyncio
async def test_huggingface_with_timeout() -> None:
    api_key = "test-api-key"
    messages = [{"role": "user", "content": "Hello"}]
    with mock_huggingface_provider() as mock_huggingface:
        provider = HuggingfaceProvider(api_key=api_key, timeout=10)
        await provider._acompletion(CompletionParams(model_id="model-id", messages=messages, max_tokens=100))
        mock_huggingface.assert_called_with(base_url=None, token=api_key, timeout=10)


@pytest.mark.asyncio
async def test_huggingface_extracts_multiple_tag_types() -> None:
    """Test that different reasoning tag formats are all extracted correctly."""
    api_key = "test-api-key"
    messages = [{"role": "user", "content": "Solve this problem"}]

    test_cases = [
        ("<think>First thought</think>\n\nAnswer", "First thought"),
        ("<thinking>Second thought</thinking>\n\nAnswer", "Second thought"),
        ("<chain_of_thought>Step by step</chain_of_thought>\n\nAnswer", "Step by step"),
    ]

    for content_with_tags, expected_reasoning in test_cases:
        with patch("any_llm.providers.huggingface.huggingface.AsyncInferenceClient") as mock_huggingface:
            async_mock = AsyncMock()
            mock_huggingface.return_value = async_mock
            async_mock.chat_completion.return_value = {
                "id": "hf-response-id",
                "created": 0,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content_with_tags, "tool_calls": None},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            }

            provider = HuggingfaceProvider(api_key=api_key)
            result = await provider._acompletion(CompletionParams(model_id="model-id", messages=messages))
            assert isinstance(result, ChatCompletion)
            assert result.choices[0].message.content == "Answer"
            assert result.choices[0].message.reasoning is not None
            assert result.choices[0].message.reasoning.content == expected_reasoning


@pytest.mark.asyncio
async def test_huggingface_extracts_think_tags_streaming() -> None:
    """Test that <think> tags split across chunks are properly extracted in streaming mode."""
    api_key = "test-api-key"
    messages = [{"role": "user", "content": "What is 2+2?"}]

    async def mock_stream() -> AsyncGenerator[Any]:
        from huggingface_hub.inference._generated.types import (  # type: ignore[attr-defined]
            ChatCompletionStreamOutput,
            ChatCompletionStreamOutputChoice,
            ChatCompletionStreamOutputDelta,
        )

        chunks = [
            "<th",
            "ink>",
            "Let me ",
            "calculate",
            " this.",
            "</think>",
            "\n\nThe ",
            "answer ",
            "is 4.",
        ]

        for i, chunk_text in enumerate(chunks):
            yield ChatCompletionStreamOutput(
                id=f"chunk-{i}",
                choices=[
                    ChatCompletionStreamOutputChoice(
                        index=0,
                        delta=ChatCompletionStreamOutputDelta(content=chunk_text, role="assistant"),
                        finish_reason="stop" if i == len(chunks) - 1 else None,
                    )
                ],
                created=0,
                model="test-model",
                system_fingerprint="test-fingerprint",
            )

    with patch("any_llm.providers.huggingface.huggingface.AsyncInferenceClient") as mock_huggingface:
        async_mock = AsyncMock()
        mock_huggingface.return_value = async_mock
        async_mock.chat_completion.return_value = mock_stream()

        provider = HuggingfaceProvider(api_key=api_key)
        result = await provider._acompletion(CompletionParams(model_id="model-id", messages=messages, stream=True))

        full_content = ""
        full_reasoning = ""
        assert hasattr(result, "__aiter__")

        async for chunk in result:
            assert isinstance(chunk, ChatCompletionChunk)
            if len(chunk.choices) > 0:
                if chunk.choices[0].delta.content:
                    full_content += chunk.choices[0].delta.content
                if chunk.choices[0].delta.reasoning:
                    full_reasoning += chunk.choices[0].delta.reasoning.content

        assert full_content.strip() == "The answer is 4."
        assert full_reasoning == "Let me calculate this."


def make_hf_chunk(
    content: str | None = None,
    role: str = "assistant",
    tool_calls: list[Any] | None = None,
    reasoning: dict[str, str] | None = None,
    finish_reason: str = "stop",
    usage: dict[str, int] | None = None,
) -> Mock:
    """Create a mock HuggingFace streaming chunk for testing."""
    delta_mock = Mock()
    delta_mock.content = content
    delta_mock.role = role
    delta_mock.tool_calls = tool_calls
    if reasoning is not None:
        delta_mock.reasoning = reasoning

    choice_mock = Mock()
    choice_mock.delta = delta_mock
    choice_mock.finish_reason = finish_reason

    usage_mock = None
    if usage:
        usage_mock = Mock()
        usage_mock.prompt_tokens = usage.get("prompt_tokens", 0)
        usage_mock.completion_tokens = usage.get("completion_tokens", 0)
        usage_mock.total_tokens = usage.get("total_tokens", 0)

    chunk_mock = Mock()
    chunk_mock.choices = [choice_mock]
    chunk_mock.created = 123456
    chunk_mock.model = "test-model"
    chunk_mock.usage = usage_mock

    return chunk_mock


def make_tool_call_mock(
    tc_id: str | None = "call-123",
    index: int | None = 0,
    name: str = "function",
    arguments: str = "{}",
    has_function: bool = True,
) -> Mock:
    """Create a mock tool call for testing."""
    tool_call_mock = Mock()
    tool_call_mock.id = tc_id
    tool_call_mock.index = index
    if has_function:
        func_mock = Mock()
        func_mock.name = name
        func_mock.arguments = arguments
        tool_call_mock.function = func_mock
    else:
        tool_call_mock.function = None
    return tool_call_mock


def test_create_openai_chunk_with_tool_calls() -> None:
    """Test streaming chunk handles tool calls correctly."""
    tool_call = make_tool_call_mock(tc_id="call-123", index=0, name="get_weather", arguments='{"location": "Paris"}')
    chunk = make_hf_chunk(
        tool_calls=[tool_call],
        finish_reason="tool_calls",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )

    result = _create_openai_chunk_from_huggingface_chunk(chunk)

    assert len(result.choices) == 1
    assert result.choices[0].delta.tool_calls is not None
    assert len(result.choices[0].delta.tool_calls) == 1
    tc = result.choices[0].delta.tool_calls[0]
    assert tc.id == "call-123"
    assert tc.index == 0
    assert tc.function is not None
    assert tc.function.name == "get_weather"
    assert tc.function.arguments == '{"location": "Paris"}'


def test_create_openai_chunk_with_tool_calls_missing_id() -> None:
    """Test streaming chunk generates id when tool call id is missing."""
    tool_call = make_tool_call_mock(tc_id=None, index=None, name="search", arguments="{}")
    chunk = make_hf_chunk(tool_calls=[tool_call], finish_reason="tool_calls")

    result = _create_openai_chunk_from_huggingface_chunk(chunk)

    assert result.choices[0].delta.tool_calls is not None
    tc = result.choices[0].delta.tool_calls[0]
    assert tc.id is not None
    assert tc.id.startswith("call_")


def test_create_openai_chunk_with_tool_calls_missing_function() -> None:
    """Test streaming chunk handles tool call without function."""
    tool_call = make_tool_call_mock(tc_id="call-456", index=0, has_function=False)
    chunk = make_hf_chunk(tool_calls=[tool_call], finish_reason="tool_calls")

    result = _create_openai_chunk_from_huggingface_chunk(chunk)

    assert result.choices[0].delta.tool_calls is not None
    tc = result.choices[0].delta.tool_calls[0]
    assert tc.function is not None
    assert tc.function.name == ""
    assert tc.function.arguments == ""


def test_create_openai_chunk_with_reasoning_dict() -> None:
    """Test streaming chunk handles reasoning as dict format."""
    chunk = make_hf_chunk(content="Answer", reasoning={"content": "Thinking..."})

    result = _create_openai_chunk_from_huggingface_chunk(chunk)

    assert result.choices[0].delta.content == "Answer"
    assert result.choices[0].delta.reasoning is not None
    assert result.choices[0].delta.reasoning.content == "Thinking..."


def test_create_openai_chunk_without_usage() -> None:
    """Test streaming chunk handles missing usage metadata."""
    chunk = make_hf_chunk(content="Hello")

    result = _create_openai_chunk_from_huggingface_chunk(chunk)

    assert result.usage is None
    assert result.choices[0].delta.content == "Hello"


# ===== OpenResponses API Tests =====


@contextmanager
def mock_huggingface_responses_provider():  # type: ignore[no-untyped-def]
    """Mock both HuggingFace inference client and OpenAI responses client."""
    with (
        patch("any_llm.providers.huggingface.huggingface.AsyncInferenceClient") as mock_hf,
        patch("any_llm.providers.huggingface.huggingface.AsyncOpenAI") as mock_openai,
    ):
        # Mock HuggingFace client
        hf_mock = AsyncMock()
        mock_hf.return_value = hf_mock

        # Mock OpenAI client for responses
        openai_mock = AsyncMock()
        mock_openai.return_value = openai_mock

        yield mock_hf, mock_openai, openai_mock


def test_huggingface_supports_responses() -> None:
    """Test that HuggingFace provider has SUPPORTS_RESPONSES=True."""
    assert HuggingfaceProvider.SUPPORTS_RESPONSES is True


def test_huggingface_openresponses_router_url() -> None:
    """Test that HuggingFace provider has the correct OpenResponses router URL."""
    assert HuggingfaceProvider.OPENRESPONSES_ROUTER_URL == "https://router.huggingface.co/v1"


@pytest.mark.asyncio
async def test_huggingface_responses_client_initialized() -> None:
    """Test that the OpenAI responses client is initialized with correct parameters."""
    api_key = "test-hf-token"

    with mock_huggingface_responses_provider() as (_, mock_openai, _):
        HuggingfaceProvider(api_key=api_key)

        mock_openai.assert_called_once_with(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )


@pytest.mark.asyncio
async def test_huggingface_aresponses_non_streaming() -> None:
    """Test non-streaming responses call."""
    api_key = "test-hf-token"
    model = "moonshotai/Kimi-K2-Instruct:groq"
    input_data = "What is the capital of France?"

    mock_response = Mock(spec=Response)
    mock_response.output_text = "The capital of France is Paris."

    with mock_huggingface_responses_provider() as (_, _, openai_mock):
        openai_mock.responses.create = AsyncMock(return_value=mock_response)

        provider = HuggingfaceProvider(api_key=api_key)
        result = await provider._aresponses(ResponsesParams(model=model, input=input_data))

        assert result == mock_response
        openai_mock.responses.create.assert_called_once()
        call_kwargs = openai_mock.responses.create.call_args.kwargs
        assert call_kwargs["model"] == model
        assert call_kwargs["input"] == input_data


@pytest.mark.asyncio
async def test_huggingface_aresponses_with_instructions() -> None:
    """Test responses call with instructions parameter."""
    api_key = "test-hf-token"
    model = "openai/gpt-oss-120b:groq"
    input_data = "Tell me a story."
    instructions = "You are a helpful assistant. Be concise."

    mock_response = Mock(spec=Response)
    mock_response.output_text = "Once upon a time..."

    with mock_huggingface_responses_provider() as (_, _, openai_mock):
        openai_mock.responses.create = AsyncMock(return_value=mock_response)

        provider = HuggingfaceProvider(api_key=api_key)
        result = await provider._aresponses(ResponsesParams(model=model, input=input_data, instructions=instructions))

        assert result == mock_response
        call_kwargs = openai_mock.responses.create.call_args.kwargs
        assert call_kwargs["instructions"] == instructions


@pytest.mark.asyncio
async def test_huggingface_aresponses_with_tools() -> None:
    """Test responses call with tools parameter."""
    api_key = "test-hf-token"
    model = "moonshotai/Kimi-K2-Instruct:groq"
    input_data = "What is the weather in Paris?"
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]

    mock_response = Mock(spec=Response)

    with mock_huggingface_responses_provider() as (_, _, openai_mock):
        openai_mock.responses.create = AsyncMock(return_value=mock_response)

        provider = HuggingfaceProvider(api_key=api_key)
        await provider._aresponses(ResponsesParams(model=model, input=input_data, tools=tools, tool_choice="auto"))

        call_kwargs = openai_mock.responses.create.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_huggingface_aresponses_streaming() -> None:
    """Test streaming responses call returns async iterator."""
    api_key = "test-hf-token"
    model = "moonshotai/Kimi-K2-Instruct:groq"
    input_data = "Tell me about AI."

    # Create mock streaming response events
    mock_events = [
        Mock(type="response.created"),
        Mock(type="response.output_text.delta", delta="AI is"),
        Mock(type="response.output_text.delta", delta=" artificial intelligence."),
        Mock(type="response.completed"),
    ]

    async def mock_stream_iterator() -> AsyncGenerator[Mock, None]:
        for event in mock_events:
            yield event

    # Create a simple async iterable mock (not a Response)
    class MockAsyncStream:
        def __aiter__(self) -> AsyncIterator[Mock]:
            return mock_stream_iterator()

    with mock_huggingface_responses_provider() as (_, _, openai_mock):
        openai_mock.responses.create = AsyncMock(return_value=MockAsyncStream())

        provider = HuggingfaceProvider(api_key=api_key)
        result = await provider._aresponses(ResponsesParams(model=model, input=input_data, stream=True))

        # Verify it returns an async iterator (not a Response)
        assert not isinstance(result, Response)

        # Consume the stream
        events_received = []
        async for event in result:
            events_received.append(event)

        assert len(events_received) == 4


@pytest.mark.asyncio
async def test_huggingface_aresponses_with_reasoning() -> None:
    """Test responses call with reasoning parameter."""
    api_key = "test-hf-token"
    model = "openai/gpt-oss-120b:groq"
    input_data = "Solve this math problem: 2 + 2"
    reasoning = {"effort": "high"}

    mock_response = Mock(spec=Response)
    mock_response.output_text = "4"

    with mock_huggingface_responses_provider() as (_, _, openai_mock):
        openai_mock.responses.create = AsyncMock(return_value=mock_response)

        provider = HuggingfaceProvider(api_key=api_key)
        await provider._aresponses(ResponsesParams(model=model, input=input_data, reasoning=reasoning))

        call_kwargs = openai_mock.responses.create.call_args.kwargs
        assert call_kwargs["reasoning"] == reasoning


@pytest.mark.asyncio
async def test_huggingface_aresponses_with_temperature() -> None:
    """Test responses call with temperature and other sampling parameters."""
    api_key = "test-hf-token"
    model = "moonshotai/Kimi-K2-Instruct:groq"
    input_data = "Generate a creative story."

    mock_response = Mock(spec=Response)

    with mock_huggingface_responses_provider() as (_, _, openai_mock):
        openai_mock.responses.create = AsyncMock(return_value=mock_response)

        provider = HuggingfaceProvider(api_key=api_key)
        await provider._aresponses(
            ResponsesParams(
                model=model,
                input=input_data,
                temperature=0.9,
                top_p=0.95,
                max_output_tokens=500,
            )
        )

        call_kwargs = openai_mock.responses.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["max_output_tokens"] == 500


@pytest.mark.asyncio
async def test_huggingface_aresponses_model_with_provider_suffix() -> None:
    """Test that model:provider syntax is passed correctly to the router."""
    api_key = "test-hf-token"
    # Model with provider suffix for routing
    model = "Qwen/Qwen2.5-VL-7B-Instruct:nebius"
    input_data = "What do you see?"

    mock_response = Mock(spec=Response)

    with mock_huggingface_responses_provider() as (_, _, openai_mock):
        openai_mock.responses.create = AsyncMock(return_value=mock_response)

        provider = HuggingfaceProvider(api_key=api_key)
        await provider._aresponses(ResponsesParams(model=model, input=input_data))

        call_kwargs = openai_mock.responses.create.call_args.kwargs
        # The model:provider syntax should be passed as-is to the router
        assert call_kwargs["model"] == model
