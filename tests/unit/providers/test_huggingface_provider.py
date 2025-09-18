from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from any_llm.providers.huggingface.huggingface import HuggingfaceProvider
from any_llm.types.completion import CompletionParams


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
