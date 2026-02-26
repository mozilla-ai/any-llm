from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from any_llm.exceptions import UnsupportedParameterError
from any_llm.types.completion import CompletionParams

groq = pytest.importorskip("groq")


@pytest.mark.asyncio
async def test_stream_with_response_format_raises() -> None:
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    provider = GroqProvider(api_key="test-api-key")

    with pytest.raises(UnsupportedParameterError, match="stream and response_format"):
        await provider._acompletion(
            CompletionParams(
                model_id="model-id",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                response_format={"type": "json_object"},
            )
        )


@pytest.mark.asyncio
async def test_unsupported_max_tool_calls_parameter() -> None:
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    provider = GroqProvider(api_key="test-api-key")

    with pytest.raises(UnsupportedParameterError):
        await provider.aresponses("test_model", "test_data", max_tool_calls=3)


@pytest.mark.asyncio
async def test_completion_with_response_format_basemodel() -> None:
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    class TestOutput(BaseModel):
        foo: str

    with (
        patch("any_llm.providers.groq.groq.AsyncGroq") as mocked_groq,
        patch("any_llm.providers.groq.groq.to_chat_completion") as mocked_to_chat_completion,
    ):
        provider = GroqProvider(api_key="test-api-key")

        mock_response = Mock()
        mocked_groq.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
        mocked_to_chat_completion.return_value = mock_response

        await provider._acompletion(
            CompletionParams(
                model_id="model-id",
                messages=[{"role": "user", "content": "Hello"}],
                response_format=TestOutput,
            ),
        )
        call_kwargs = mocked_groq.return_value.chat.completions.create.call_args[1]

        assert call_kwargs["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "TestOutput",
                "schema": {
                    "properties": {"foo": {"title": "Foo", "type": "string"}},
                    "required": ["foo"],
                    "title": "TestOutput",
                    "type": "object",
                },
            },
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning_effort", ["auto", "none"])
async def test_reasoning_effort_filtered_out(reasoning_effort: str) -> None:
    """Test that reasoning_effort 'auto' and 'none' are filtered from Groq API calls."""
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    with (
        patch("any_llm.providers.groq.groq.AsyncGroq") as mocked_groq,
        patch("any_llm.providers.groq.groq.to_chat_completion") as mocked_to_chat_completion,
    ):
        provider = GroqProvider(api_key="test-api-key")

        mock_response = Mock()
        mocked_groq.return_value.chat.completions.create = AsyncMock(return_value=mock_response)
        mocked_to_chat_completion.return_value = mock_response

        await provider._acompletion(
            CompletionParams(
                model_id="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Hello"}],
                reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            ),
        )

        call_kwargs = mocked_groq.return_value.chat.completions.create.call_args[1]
        assert "reasoning_effort" not in call_kwargs


def test_to_chat_completion_extracts_cached_tokens() -> None:
    """Test that cached tokens from Groq usage are extracted into prompt_tokens_details."""
    from groq.types.chat import ChatCompletion as GroqChatCompletion

    from any_llm.providers.groq.utils import to_chat_completion

    response = GroqChatCompletion.model_validate(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "llama-3.3-70b-versatile",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 4641,
                "completion_tokens": 1817,
                "total_tokens": 6458,
                "prompt_tokens_details": {"cached_tokens": 4608},
            },
        }
    )

    result = to_chat_completion(response)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 4641
    assert result.usage.completion_tokens == 1817
    assert result.usage.total_tokens == 6458
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 4608


def test_to_chat_completion_without_cached_tokens() -> None:
    """Test that prompt_tokens_details is None when no cached tokens are present."""
    from groq.types.chat import ChatCompletion as GroqChatCompletion

    from any_llm.providers.groq.utils import to_chat_completion

    response = GroqChatCompletion.model_validate(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "llama-3.3-70b-versatile",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
    )

    result = to_chat_completion(response)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 150
    assert result.usage.prompt_tokens_details is None


def test_streaming_chunk_extracts_cached_tokens() -> None:
    """Test that streaming chunks correctly extract cached tokens from Groq usage."""
    from groq.types.chat import ChatCompletionChunk as GroqChatCompletionChunk

    from any_llm.providers.groq.utils import _create_openai_chunk_from_groq_chunk

    chunk = GroqChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "llama-3.3-70b-versatile",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 4641,
                "completion_tokens": 1817,
                "total_tokens": 6458,
                "prompt_tokens_details": {"cached_tokens": 4608},
            },
        }
    )

    result = _create_openai_chunk_from_groq_chunk(chunk)

    assert result.usage is not None
    assert result.usage.prompt_tokens == 4641
    assert result.usage.prompt_tokens_details is not None
    assert result.usage.prompt_tokens_details.cached_tokens == 4608
