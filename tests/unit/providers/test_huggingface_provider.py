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
