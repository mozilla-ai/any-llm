import pytest

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.cerebras.cerebras import CerebrasProvider
from any_llm.providers.cerebras.utils import _convert_response, _create_openai_chunk_from_cerebras_chunk


@pytest.mark.asyncio
async def test_stream_with_response_format_raises() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    provider = CerebrasProvider(api_key=api_key)

    chunks = provider._stream_completion_async(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    with pytest.raises(UnsupportedParameterError):
        async for _ in chunks:
            pass


def test_convert_response_extracts_reasoning() -> None:
    response_data = {
        "id": "test-id",
        "model": "llama-3.3-70b",
        "created": 1234567890,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "reasoning": "The user asked me to say hello, so I will respond with a greeting.",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    result = _convert_response(response_data)

    assert result.choices[0].message.content == "Hello!"
    assert result.choices[0].message.reasoning is not None
    assert (
        result.choices[0].message.reasoning.content
        == "The user asked me to say hello, so I will respond with a greeting."
    )


def test_convert_response_without_reasoning() -> None:
    response_data = {
        "id": "test-id",
        "model": "llama-3.3-70b",
        "created": 1234567890,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "tool_calls": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    result = _convert_response(response_data)

    assert result.choices[0].message.content == "Hello!"
    assert result.choices[0].message.reasoning is None


def test_convert_chunk_extracts_reasoning() -> None:
    from unittest.mock import Mock

    mock_chunk = Mock()
    mock_chunk.id = "test-chunk-id"
    mock_chunk.model = "llama-3.3-70b"
    mock_chunk.created = 1234567890

    mock_delta = Mock()
    mock_delta.content = "Hello!"
    mock_delta.role = "assistant"
    mock_delta.reasoning = "Thinking about the greeting..."
    mock_delta.tool_calls = None

    mock_choice = Mock()
    mock_choice.delta = mock_delta
    mock_choice.finish_reason = None

    mock_chunk.choices = [mock_choice]
    mock_chunk.usage = None

    result = _create_openai_chunk_from_cerebras_chunk(mock_chunk)

    assert result.choices[0].delta.content == "Hello!"
    assert result.choices[0].delta.reasoning is not None
    assert result.choices[0].delta.reasoning.content == "Thinking about the greeting..."


def test_convert_chunk_without_reasoning() -> None:
    from unittest.mock import Mock

    mock_chunk = Mock()
    mock_chunk.id = "test-chunk-id"
    mock_chunk.model = "llama-3.3-70b"
    mock_chunk.created = 1234567890

    mock_delta = Mock()
    mock_delta.content = "Hello!"
    mock_delta.role = "assistant"
    mock_delta.reasoning = None
    mock_delta.tool_calls = None

    mock_choice = Mock()
    mock_choice.delta = mock_delta
    mock_choice.finish_reason = None

    mock_chunk.choices = [mock_choice]
    mock_chunk.usage = None

    result = _create_openai_chunk_from_cerebras_chunk(mock_chunk)

    assert result.choices[0].delta.content == "Hello!"
    assert "reasoning" not in result.choices[0].delta.model_dump(exclude_none=True)
