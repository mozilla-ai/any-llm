from collections.abc import AsyncGenerator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from any_llm.providers.huggingface.huggingface import HuggingfaceProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams


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
                        delta=ChatCompletionStreamOutputDelta(content=chunk_text, role="assistant" if i == 0 else None),
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
