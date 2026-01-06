from collections.abc import AsyncIterable, AsyncIterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.providers.watsonx.utils import _convert_streaming_chunk
from any_llm.providers.watsonx.watsonx import WatsonxProvider
from any_llm.types.completion import CompletionParams


@contextmanager
def mock_watsonx_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.watsonx.watsonx.ModelInference") as mock_model_inference,
        patch("any_llm.providers.watsonx.watsonx._convert_response") as mock_convert_response,
    ):
        mock_model_instance = MagicMock()
        mock_model_inference.return_value = mock_model_instance

        mock_watsonx_response = {"choices": [{"message": {"content": "Hello"}}]}
        mock_model_instance.achat = AsyncMock(return_value=mock_watsonx_response)

        mock_openai_response = MagicMock()
        mock_convert_response.return_value = mock_openai_response

        yield mock_model_instance, mock_convert_response, mock_model_inference


@contextmanager
def mock_watsonx_streaming_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.watsonx.watsonx.ModelInference") as mock_model_inference,
        patch("any_llm.providers.watsonx.watsonx._convert_streaming_chunk") as mock_convert_streaming_chunk,
    ):
        mock_model_instance = MagicMock()
        mock_model_inference.return_value = mock_model_instance

        mock_watsonx_chunk1 = {"choices": [{"delta": {"content": "Hello"}}]}
        mock_watsonx_chunk2 = {"choices": [{"delta": {"content": " World"}}]}

        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            yield mock_watsonx_chunk1
            yield mock_watsonx_chunk2

        mock_model_instance.achat_stream = AsyncMock(return_value=mock_stream())

        mock_openai_chunk1 = MagicMock()
        mock_openai_chunk2 = MagicMock()
        mock_convert_streaming_chunk.side_effect = [mock_openai_chunk1, mock_openai_chunk2]

        yield mock_model_instance, mock_convert_streaming_chunk, mock_model_inference


@pytest.mark.asyncio
async def test_watsonx_non_streaming() -> None:
    api_key = "test-api-key"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_watsonx_provider() as (mock_model_instance, mock_convert_response, mock_model_inference):
        provider = WatsonxProvider(api_key=api_key)
        result = await provider._acompletion(CompletionParams(model_id="test-model", messages=messages))

        mock_model_inference.assert_called_once()
        call_kwargs = mock_model_inference.call_args[1]
        assert call_kwargs["model_id"] == "test-model"

        mock_model_instance.achat.assert_called_once_with(messages=messages, params={})

        mock_convert_response.assert_called_once()

        assert result == mock_convert_response.return_value


@pytest.mark.asyncio
async def test_watsonx_streaming() -> None:
    api_key = "test-api-key"
    messages = [{"role": "user", "content": "Hello"}]

    with mock_watsonx_streaming_provider() as (mock_model_instance, mock_convert_streaming_chunk, mock_model_inference):
        provider = WatsonxProvider(api_key=api_key)
        result = await provider._acompletion(CompletionParams(model_id="test-model", messages=messages, stream=True))

        mock_model_inference.assert_called_once()
        call_kwargs = mock_model_inference.call_args[1]
        assert call_kwargs["model_id"] == "test-model"

        result_list = []
        assert isinstance(result, AsyncIterable)
        async for chunk in result:
            result_list.append(chunk)

        mock_model_instance.achat_stream.assert_called_once_with(messages=messages, params={})

        assert mock_convert_streaming_chunk.call_count == 2

        assert len(result_list) == 2
        assert result_list is not None


def test_watsonx_SUPPORTS_COMPLETION_STREAMING() -> None:
    """Test that WatsonxProvider correctly advertises streaming support."""
    provider = WatsonxProvider(api_key="test-key")
    assert provider.SUPPORTS_COMPLETION_STREAMING is True


def test_convert_streaming_chunk_with_tool_calls() -> None:
    """Test streaming chunk conversion with tool calls."""
    chunk = {
        "created": 123456,
        "model": "test-model",
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-123",
                            "index": 0,
                            "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    result = _convert_streaming_chunk(chunk)

    assert len(result.choices) == 1
    assert result.choices[0].delta.tool_calls is not None
    assert len(result.choices[0].delta.tool_calls) == 1
    tool_call = result.choices[0].delta.tool_calls[0]
    assert tool_call.id == "call-123"
    assert tool_call.index == 0
    assert tool_call.function is not None
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"location": "Paris"}'


def test_convert_streaming_chunk_with_multiple_tool_calls() -> None:
    """Test streaming chunk conversion with multiple tool calls."""
    chunk = {
        "created": 123456,
        "model": "test-model",
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call-1", "index": 0, "function": {"name": "func_a", "arguments": "{}"}},
                        {"id": "call-2", "index": 1, "function": {"name": "func_b", "arguments": "{}"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    result = _convert_streaming_chunk(chunk)

    assert result.choices[0].delta.tool_calls is not None
    assert len(result.choices[0].delta.tool_calls) == 2
    assert result.choices[0].delta.tool_calls[0].function is not None
    assert result.choices[0].delta.tool_calls[0].function.name == "func_a"
    assert result.choices[0].delta.tool_calls[1].function is not None
    assert result.choices[0].delta.tool_calls[1].function.name == "func_b"


def test_convert_streaming_chunk_with_tool_calls_missing_fields() -> None:
    """Test streaming chunk handles tool calls with missing optional fields."""
    chunk = {
        "created": 123456,
        "model": "test-model",
        "choices": [
            {
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "func", "arguments": "{}"}},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    result = _convert_streaming_chunk(chunk)

    assert result.choices[0].delta.tool_calls is not None
    tool_call = result.choices[0].delta.tool_calls[0]
    assert tool_call.id is not None
    assert tool_call.id.startswith("call_")


def test_convert_streaming_chunk_with_text_content() -> None:
    """Test streaming chunk conversion with text content."""
    chunk = {
        "created": 123456,
        "model": "test-model",
        "choices": [
            {
                "delta": {"content": "Hello world", "role": "assistant"},
                "finish_reason": "stop",
            }
        ],
    }

    result = _convert_streaming_chunk(chunk)

    assert result.choices[0].delta.content == "Hello world"
    assert result.choices[0].delta.role == "assistant"
    assert result.choices[0].finish_reason == "stop"


def test_convert_streaming_chunk_with_usage() -> None:
    """Test streaming chunk conversion with usage metadata."""
    chunk = {
        "created": 123456,
        "model": "test-model",
        "choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    result = _convert_streaming_chunk(chunk)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 5
    assert result.usage.total_tokens == 15


def test_convert_streaming_chunk_without_usage() -> None:
    """Test streaming chunk conversion without usage metadata."""
    chunk = {
        "created": 123456,
        "model": "test-model",
        "choices": [{"delta": {"content": "Hi"}, "finish_reason": "stop"}],
    }

    result = _convert_streaming_chunk(chunk)

    assert result.usage is None


def test_convert_streaming_chunk_empty_choices() -> None:
    """Test streaming chunk conversion with empty choices."""
    chunk = {"created": 123456, "model": "test-model", "choices": []}

    result = _convert_streaming_chunk(chunk)

    assert result.choices == []
