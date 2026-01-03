import time
from contextlib import contextmanager
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.providers.azureopenai.azureopenai import AzureopenaiProvider
from any_llm.types.completion import CompletionParams


@contextmanager
def mock_azureopenai_provider():  # type: ignore[no-untyped-def]
    with (
        patch("any_llm.providers.openai.base.AsyncOpenAI") as mock_openai_client,
    ):
        mock_client_instance = MagicMock()
        mock_openai_client.return_value = mock_client_instance

        mock_response = MagicMock()
        mock_client_instance.chat.completions.create = AsyncMock(return_value=mock_response)

        yield mock_client_instance, mock_openai_client


@pytest.mark.asyncio
async def test_azureopenai_default_query_with_existing_kwargs() -> None:
    """Test that AzureopenaiProvider preserves existing kwargs while adding default_query."""
    api_key = "test-api-key"
    api_base = "https://test.openai.azure.com"
    custom_timeout = 30

    messages = [{"role": "user", "content": "Hello"}]

    with mock_azureopenai_provider() as (mock_client, mock_openai_client):
        provider = AzureopenaiProvider(api_key=api_key, api_base=api_base, timeout=custom_timeout)
        await provider._acompletion(CompletionParams(model_id="gpt-4", messages=messages))

        mock_openai_client.assert_called_once()
        call_args = mock_openai_client.call_args
        assert call_args is not None
        _, kwargs = call_args
        assert "default_query" in kwargs
        assert kwargs["default_query"] == {"api-version": "preview"}
        assert kwargs["timeout"] == custom_timeout

        mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_init_client_with_api_version():
    api_key = "test-azure-api-key"
    api_base = "https://example.openai.azure.com/"
    api_version = "2024-02-01"

    # Patch AsyncAzureOpenAI, because _init_client instantiates it
    with patch("any_llm.providers.azureopenai.azureopenai.AsyncAzureOpenAI") as mock_async_azure:
        mock_instance = MagicMock()
        mock_async_azure.return_value = mock_instance

        provider = AzureopenaiProvider(api_key=api_key, api_base=api_base, **{"api-version": api_version})

        # Check AsyncAzureOpenAI was called with expected kwargs
        mock_async_azure.assert_called_once()
        _, kwargs = mock_async_azure.call_args
        assert kwargs["azure_endpoint"] == api_base
        assert kwargs["api_key"] == api_key
        assert kwargs["api_version"] == api_version

        # Ensure the provider.client points to the mocked instance
        assert provider.client is mock_instance


@pytest.mark.asyncio
async def test_init_client_default_api_version():
    """
    Verifies fallback to "preview" when api-version is not provided.
    """
    api_key = "test-api-key"
    api_base = "https://test.openai.azure.com"

    with patch("any_llm.providers.azureopenai.azureopenai.AsyncAzureOpenAI") as mock_async_azure:
        mock_instance = MagicMock()
        mock_async_azure.return_value = mock_instance

        # Initialize provider (calls _init_client internally)
        provider = AzureopenaiProvider(api_key=api_key, api_base=api_base)

        # Assert AsyncAzureOpenAI was called exactly once
        mock_async_azure.assert_called_once()
        _, kwargs = mock_async_azure.call_args
        assert kwargs["azure_endpoint"] == api_base
        assert kwargs["api_key"] == api_key
        assert kwargs["api_version"] == "preview"


async def verify_model_response(generator, expected_chunks: List[dict]) -> None:
    """
    Consumes an async generator (from _acompletion or similar)
    and verifies that the yielded chunks match the expected output.

    Args:
        generator: Async generator from the model/provider.
        expected_chunks: List of dictionaries, each representing a chunk.
                         Each dict should have at least "role" and "content".
    """
    # Collect all chunks from the generator
    result_list = [chunk async for chunk in generator]

    # Check number of chunks matches
    assert len(result_list) == len(expected_chunks), (
        f"Expected {len(expected_chunks)} chunks, got {len(result_list)}"
    )

    # Compare each chunk
    for i, (result, expected) in enumerate(zip(result_list, expected_chunks)):
        # If result is a ChatCompletionChunk or dict-like object
        if hasattr(result, "dict"):
            result_data = result.dict()
        else:
            result_data = result

        assert result_data["role"] == expected["role"], f"Chunk {i} role mismatch"
        assert result_data["content"] == expected["content"], f"Chunk {i} content mismatch"


@pytest.mark.asyncio
async def test_acompletion_with_valid_mocked_chunk():
    api_key = "test-api-key"
    api_base = "https://example.openai.azure.com/"
    messages = [{"role": "user", "content": "Hello"}]

    # Correct mock chunk
    mock_chunk = {
        "id": "test-id",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "gpt-4",
        "choices": [
            {
                "delta": {"content": "Hello"},
                "index": 0,
                "role": "assistant",
                "finish_reason": None
            }
        ]
    }

    async def async_gen():
        yield mock_chunk

    # Patch AsyncAzureOpenAI
    with patch("any_llm.providers.azureopenai.azureopenai.AsyncAzureOpenAI") as mock_async_azure:
        mock_client_instance = MagicMock()
        mock_async_azure.return_value = mock_client_instance
        mock_client_instance.chat.completions.create = AsyncMock(return_value=async_gen())

        provider = AzureopenaiProvider(api_key=api_key, api_base=api_base)

        gen = await provider._acompletion(CompletionParams(model_id="gpt-4", messages=messages))

        # Consume async generator
        result_list = [chunk async for chunk in gen]

        # Verify result
        assert len(result_list) == 1
        chunk = result_list[0]

        assert chunk.id == "test-id"
        assert chunk.model == "gpt-4"

        # Access via attributes
        choice = chunk.choices[0]
        assert choice.delta.content == "Hello"
        assert choice.role == "assistant"
