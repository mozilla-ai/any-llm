"""Tests for batch methods on the AnyLLM base class and the top-level api.py functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.any_llm import AnyLLM
from any_llm.types.batch import BatchResult, BatchResultItem


@pytest.mark.asyncio
async def test_base_aretrieve_batch_results_not_supported() -> None:
    """Test that _aretrieve_batch_results raises NotImplementedError when SUPPORTS_BATCH is False."""
    mock_provider = MagicMock(spec=AnyLLM)
    mock_provider.SUPPORTS_BATCH = False

    with pytest.raises(NotImplementedError, match="doesn't support batch"):
        await AnyLLM._aretrieve_batch_results(mock_provider, "batch-123")


@pytest.mark.asyncio
async def test_base_aretrieve_batch_results_not_implemented() -> None:
    """Test that _aretrieve_batch_results raises NotImplementedError when subclass does not override."""
    mock_provider = MagicMock(spec=AnyLLM)
    mock_provider.SUPPORTS_BATCH = True

    with pytest.raises(NotImplementedError, match="Subclasses must implement"):
        await AnyLLM._aretrieve_batch_results(mock_provider, "batch-123")


@pytest.mark.asyncio
async def test_base_acreate_batch_not_supported() -> None:
    """Test that _acreate_batch raises NotImplementedError when SUPPORTS_BATCH is False."""
    from any_llm.providers.openai.openai import OpenaiProvider

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = OpenaiProvider(api_key="test-key")
        provider.SUPPORTS_BATCH = False  # type: ignore[misc]

        with pytest.raises(NotImplementedError, match="does not support batch"):
            await provider._acreate_batch("input.jsonl", "/v1/chat/completions")


@pytest.mark.asyncio
async def test_base_aretrieve_batch_not_supported() -> None:
    """Test that _aretrieve_batch raises NotImplementedError when SUPPORTS_BATCH is False."""
    from any_llm.providers.openai.openai import OpenaiProvider

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = OpenaiProvider(api_key="test-key")
        provider.SUPPORTS_BATCH = False  # type: ignore[misc]

        with pytest.raises(NotImplementedError, match="does not support batch"):
            await provider._aretrieve_batch("batch-123")


@pytest.mark.asyncio
async def test_base_acancel_batch_not_supported() -> None:
    """Test that _acancel_batch raises NotImplementedError when SUPPORTS_BATCH is False."""
    from any_llm.providers.openai.openai import OpenaiProvider

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = OpenaiProvider(api_key="test-key")
        provider.SUPPORTS_BATCH = False  # type: ignore[misc]

        with pytest.raises(NotImplementedError, match="does not support batch"):
            await provider._acancel_batch("batch-123")


@pytest.mark.asyncio
async def test_base_alist_batches_not_supported() -> None:
    """Test that _alist_batches raises NotImplementedError when SUPPORTS_BATCH is False."""
    from any_llm.providers.openai.openai import OpenaiProvider

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = OpenaiProvider(api_key="test-key")
        provider.SUPPORTS_BATCH = False  # type: ignore[misc]

        with pytest.raises(NotImplementedError, match="does not support batch"):
            await provider._alist_batches()


@pytest.mark.asyncio
async def test_aretrieve_batch_results_delegates_to_private() -> None:
    """Test that aretrieve_batch_results calls _aretrieve_batch_results on the provider."""
    from any_llm.providers.openai.openai import OpenaiProvider

    expected = BatchResult(results=[BatchResultItem(custom_id="req-1")])
    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = OpenaiProvider(api_key="test-key")
        provider._aretrieve_batch_results = AsyncMock(return_value=expected)  # type: ignore[method-assign]

        result = await provider.aretrieve_batch_results("batch-123")

    assert result is expected
    provider._aretrieve_batch_results.assert_called_once_with("batch-123")


def test_retrieve_batch_results_sync_delegates() -> None:
    """Test that retrieve_batch_results (sync) delegates to aretrieve_batch_results."""
    from any_llm.providers.openai.openai import OpenaiProvider

    expected = BatchResult(results=[BatchResultItem(custom_id="req-1")])
    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = OpenaiProvider(api_key="test-key")
        provider.aretrieve_batch_results = AsyncMock(return_value=expected)  # type: ignore[method-assign]

        result = provider.retrieve_batch_results("batch-123")

    assert result is expected
    provider.aretrieve_batch_results.assert_called_once_with("batch-123")


@pytest.mark.asyncio
async def test_api_aretrieve_batch_results() -> None:
    """Test api.aretrieve_batch_results creates provider and delegates."""
    from any_llm import api

    mock_result = BatchResult(results=[BatchResultItem(custom_id="req-1")])
    mock_provider = MagicMock()
    mock_provider.aretrieve_batch_results = AsyncMock(return_value=mock_result)

    with patch.object(api.AnyLLM, "create", return_value=mock_provider):
        result = await api.aretrieve_batch_results("openai", "batch-123", api_key="test-key")

    assert isinstance(result, BatchResult)
    assert result.results[0].custom_id == "req-1"
    mock_provider.aretrieve_batch_results.assert_called_once_with("batch-123")


def test_api_retrieve_batch_results() -> None:
    """Test api.retrieve_batch_results creates provider and delegates."""
    from any_llm import api

    mock_result = BatchResult(results=[BatchResultItem(custom_id="req-1")])
    mock_provider = MagicMock()
    mock_provider.retrieve_batch_results.return_value = mock_result

    with patch.object(api.AnyLLM, "create", return_value=mock_provider):
        result = api.retrieve_batch_results("openai", "batch-123", api_key="test-key")

    assert isinstance(result, BatchResult)
    mock_provider.retrieve_batch_results.assert_called_once_with("batch-123")


@pytest.mark.asyncio
async def test_api_acreate_batch() -> None:
    """Test api.acreate_batch creates provider and delegates."""
    from any_llm import api

    mock_batch = MagicMock()
    mock_provider = MagicMock()
    mock_provider.acreate_batch = AsyncMock(return_value=mock_batch)

    with patch.object(api.AnyLLM, "create", return_value=mock_provider):
        result = await api.acreate_batch("openai", "input.jsonl", "/v1/chat/completions", api_key="test-key")

    assert result == mock_batch


def test_api_create_batch() -> None:
    """Test api.create_batch creates provider and delegates."""
    from any_llm import api

    mock_batch = MagicMock()
    mock_provider = MagicMock()
    mock_provider.create_batch.return_value = mock_batch

    with patch.object(api.AnyLLM, "create", return_value=mock_provider):
        result = api.create_batch(
            "openai", input_file_path="input.jsonl", endpoint="/v1/chat/completions", api_key="test-key"
        )

    assert result == mock_batch


@pytest.mark.asyncio
async def test_api_aretrieve_batch() -> None:
    """Test api.aretrieve_batch creates provider and delegates."""
    from any_llm import api

    mock_batch = MagicMock()
    mock_provider = MagicMock()
    mock_provider.aretrieve_batch = AsyncMock(return_value=mock_batch)

    with patch.object(api.AnyLLM, "create", return_value=mock_provider):
        result = await api.aretrieve_batch("openai", "batch-123", api_key="test-key")

    assert result == mock_batch


def test_api_retrieve_batch() -> None:
    """Test api.retrieve_batch creates provider and delegates."""
    from any_llm import api

    mock_batch = MagicMock()
    mock_provider = MagicMock()
    mock_provider.retrieve_batch.return_value = mock_batch

    with patch.object(api.AnyLLM, "create", return_value=mock_provider):
        result = api.retrieve_batch("openai", "batch-123", api_key="test-key")

    assert result == mock_batch


@pytest.mark.asyncio
async def test_api_acancel_batch() -> None:
    """Test api.acancel_batch creates provider and delegates."""
    from any_llm import api

    mock_batch = MagicMock()
    mock_provider = MagicMock()
    mock_provider.acancel_batch = AsyncMock(return_value=mock_batch)

    with patch.object(api.AnyLLM, "create", return_value=mock_provider):
        result = await api.acancel_batch("openai", "batch-123", api_key="test-key")

    assert result == mock_batch


def test_api_cancel_batch() -> None:
    """Test api.cancel_batch creates provider and delegates."""
    from any_llm import api

    mock_batch = MagicMock()
    mock_provider = MagicMock()
    mock_provider.cancel_batch.return_value = mock_batch

    with patch.object(api.AnyLLM, "create", return_value=mock_provider):
        result = api.cancel_batch("openai", "batch-123", api_key="test-key")

    assert result == mock_batch


@pytest.mark.asyncio
async def test_api_alist_batches() -> None:
    """Test api.alist_batches creates provider and delegates."""
    from any_llm import api

    mock_batches = [MagicMock()]
    mock_provider = MagicMock()
    mock_provider.alist_batches = AsyncMock(return_value=mock_batches)

    with patch.object(api.AnyLLM, "create", return_value=mock_provider):
        result = await api.alist_batches("openai", api_key="test-key")

    assert result == mock_batches


def test_api_list_batches() -> None:
    """Test api.list_batches creates provider and delegates."""
    from any_llm import api

    mock_batches = [MagicMock()]
    mock_provider = MagicMock()
    mock_provider.list_batches.return_value = mock_batches

    with patch.object(api.AnyLLM, "create", return_value=mock_provider):
        result = api.list_batches("openai", api_key="test-key")

    assert result == mock_batches
