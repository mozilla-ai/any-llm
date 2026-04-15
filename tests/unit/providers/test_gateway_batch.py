import json
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.exceptions import BatchNotCompleteError, InvalidRequestError, ProviderError
from any_llm.types.batch import BatchResult


def _create_gateway_provider() -> "GatewayProvider":  # type: ignore[name-defined]  # noqa: F821
    """Helper to create a GatewayProvider with mocked OpenAI client."""
    from any_llm.providers.gateway.gateway import GatewayProvider

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        return GatewayProvider(api_key="test-key", api_base="https://gateway.example.com/v1/")


@pytest.mark.asyncio
async def test_acreate_batch_sends_json_body() -> None:
    """Test that _acreate_batch sends JSON body to /v1/batches, not a file upload."""
    provider = _create_gateway_provider()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "batch-gw-123",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "status": "validating",
        "created_at": 1700000000,
        "completion_window": "24h",
        "request_counts": {"total": 1, "completed": 0, "failed": 0},
        "input_file_id": "",
        "output_file_id": None,
        "error_file_id": None,
        "metadata": None,
    }
    provider.client._client.post = AsyncMock(return_value=mock_response)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(
            json.dumps(
                {
                    "custom_id": "req-1",
                    "body": {
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                }
            )
            + "\n"
        )
        tmp_path = f.name

    try:
        result = await provider._acreate_batch(
            input_file_path=tmp_path,
            endpoint="/v1/chat/completions",
        )

        assert result.id == "batch-gw-123"
        provider.client._client.post.assert_called_once()
        call_args = provider.client._client.post.call_args
        assert "batches" in str(call_args[0][0])
        body = call_args[1]["json"]
        assert "requests" in body
        assert body["model"] == "gpt-4"
    finally:
        import os

        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_aretrieve_batch_sends_provider_query_param() -> None:
    """Test that _aretrieve_batch sends ?provider= query param."""
    provider = _create_gateway_provider()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "batch-123",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "status": "completed",
        "created_at": 1700000000,
        "completion_window": "24h",
        "request_counts": {"total": 1, "completed": 1, "failed": 0},
        "input_file_id": "file-input-123",
        "output_file_id": None,
        "error_file_id": None,
        "metadata": None,
    }
    provider.client._client.get = AsyncMock(return_value=mock_response)

    await provider._aretrieve_batch("batch-123", provider_name="openai")

    provider.client._client.get.assert_called_once()
    call_kwargs = provider.client._client.get.call_args[1]
    assert call_kwargs["params"]["provider"] == "openai"


@pytest.mark.asyncio
async def test_acancel_batch_sends_provider_query_param() -> None:
    """Test that _acancel_batch sends ?provider= query param."""
    provider = _create_gateway_provider()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "batch-123",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "status": "cancelling",
        "created_at": 1700000000,
        "completion_window": "24h",
        "request_counts": {"total": 1, "completed": 0, "failed": 0},
        "input_file_id": "file-input-123",
        "output_file_id": None,
        "error_file_id": None,
        "metadata": None,
    }
    provider.client._client.post = AsyncMock(return_value=mock_response)

    await provider._acancel_batch("batch-123", provider_name="openai")

    provider.client._client.post.assert_called_once()
    call_kwargs = provider.client._client.post.call_args[1]
    assert call_kwargs["params"]["provider"] == "openai"


@pytest.mark.asyncio
async def test_alist_batches_sends_query_params() -> None:
    """Test that _alist_batches sends ?provider=, ?after=, ?limit= query params."""
    provider = _create_gateway_provider()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": []}
    provider.client._client.get = AsyncMock(return_value=mock_response)

    await provider._alist_batches(after="batch-100", limit=5, provider_name="mistral")

    provider.client._client.get.assert_called_once()
    call_kwargs = provider.client._client.get.call_args[1]
    assert call_kwargs["params"]["provider"] == "mistral"
    assert call_kwargs["params"]["after"] == "batch-100"
    assert call_kwargs["params"]["limit"] == 5


@pytest.mark.asyncio
async def test_aretrieve_batch_results_sends_provider_query_param() -> None:
    """Test that _aretrieve_batch_results sends ?provider= and deserializes BatchResult."""
    provider = _create_gateway_provider()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "custom_id": "req-1",
                "result": {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "gpt-4",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                },
                "error": None,
            }
        ]
    }
    provider.client._client.get = AsyncMock(return_value=mock_response)

    result = await provider._aretrieve_batch_results("batch-123", provider_name="openai")

    assert isinstance(result, BatchResult)
    assert len(result.results) == 1
    assert result.results[0].custom_id == "req-1"
    assert result.results[0].result is not None
    assert result.results[0].result.id == "chatcmpl-1"

    provider.client._client.get.assert_called_once()
    call_kwargs = provider.client._client.get.call_args[1]
    assert call_kwargs["params"]["provider"] == "openai"


@pytest.mark.asyncio
async def test_acreate_batch_404_produces_upgrade_message() -> None:
    """Test that 404 on /v1/batches produces 'upgrade your gateway' error."""
    provider = _create_gateway_provider()

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.url = "https://gateway.example.com/v1/batches"
    mock_response.json.return_value = {"detail": "Not Found"}
    provider.client._client.post = AsyncMock(return_value=mock_response)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(
            json.dumps(
                {
                    "custom_id": "req-1",
                    "body": {"model": "gpt-4", "messages": []},
                }
            )
            + "\n"
        )
        tmp_path = f.name

    try:
        with pytest.raises(ProviderError, match="does not support batch operations"):
            await provider._acreate_batch(
                input_file_path=tmp_path,
                endpoint="/v1/chat/completions",
            )
    finally:
        import os

        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_aretrieve_batch_results_409_produces_batch_not_complete_error() -> None:
    """Test that 409 on /v1/batches/{id}/results produces BatchNotCompleteError."""
    provider = _create_gateway_provider()

    mock_response = MagicMock()
    mock_response.status_code = 409
    mock_response.json.return_value = {"detail": "Batch not complete"}
    provider.client._client.get = AsyncMock(return_value=mock_response)

    with pytest.raises(BatchNotCompleteError) as exc_info:
        await provider._aretrieve_batch_results("batch-pending", provider_name="openai")

    assert exc_info.value.batch_id == "batch-pending"


@pytest.mark.asyncio
async def test_aretrieve_batch_missing_provider_name_raises() -> None:
    """Test that missing provider_name raises InvalidRequestError."""
    provider = _create_gateway_provider()

    with pytest.raises(InvalidRequestError, match="provider_name is required"):
        await provider._aretrieve_batch("batch-123")


@pytest.mark.asyncio
async def test_acancel_batch_missing_provider_name_raises() -> None:
    """Test that missing provider_name raises InvalidRequestError."""
    provider = _create_gateway_provider()

    with pytest.raises(InvalidRequestError, match="provider_name is required"):
        await provider._acancel_batch("batch-123")


@pytest.mark.asyncio
async def test_alist_batches_missing_provider_name_raises() -> None:
    """Test that missing provider_name raises InvalidRequestError."""
    provider = _create_gateway_provider()

    with pytest.raises(InvalidRequestError, match="provider_name is required"):
        await provider._alist_batches()


@pytest.mark.asyncio
async def test_aretrieve_batch_results_missing_provider_name_raises() -> None:
    """Test that missing provider_name raises InvalidRequestError."""
    provider = _create_gateway_provider()

    with pytest.raises(InvalidRequestError, match="provider_name is required"):
        await provider._aretrieve_batch_results("batch-123")


@pytest.mark.asyncio
async def test_handle_batch_http_error_404_with_batches_url() -> None:
    """Test _handle_batch_http_error on 404 for a batches URL."""
    provider = _create_gateway_provider()

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.url = "https://gateway.example.com/v1/batches/batch-123"
    mock_response.json.return_value = {"detail": "Not found"}

    with pytest.raises(ProviderError, match="does not support batch operations"):
        provider._handle_batch_http_error(mock_response)
