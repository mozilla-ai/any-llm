import json
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.exceptions import InvalidRequestError
from any_llm.types.batch import BatchResult

pytest.importorskip("otari")

from any_llm.providers.otari.otari import OtariProvider


def _mock_otari_client() -> MagicMock:
    client = MagicMock()
    client.platform_mode = False
    client.openai = AsyncMock()
    client.create_batch = AsyncMock()
    client.retrieve_batch = AsyncMock()
    client.cancel_batch = AsyncMock()
    client.list_batches = AsyncMock()
    client.retrieve_batch_results = AsyncMock()
    return client


def _build_provider(mocked_client: MagicMock) -> OtariProvider:
    with patch("any_llm.providers.otari.otari.OtariClient", return_value=mocked_client):
        return OtariProvider(api_base="https://otari.example.com")


@pytest.mark.asyncio
async def test_otari_create_batch_uses_sdk_client() -> None:
    mocked_client = _mock_otari_client()
    mocked_client.create_batch.return_value = {
        "id": "batch-123",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "input-file-123",
        "status": "validating",
        "created_at": 1700000000,
        "completion_window": "24h",
        "request_counts": {"total": 1, "completed": 0, "failed": 0},
    }
    provider = _build_provider(mocked_client)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"custom_id": "req-1", "body": {"model": "gpt-4", "messages": []}}) + "\n")
        tmp_path = f.name

    try:
        result = await provider._acreate_batch(input_file_path=tmp_path, endpoint="/v1/chat/completions")
    finally:
        import os

        os.unlink(tmp_path)

    assert result.id == "batch-123"
    mocked_client.create_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_otari_create_batch_rejects_unsupported_endpoint() -> None:
    provider = _build_provider(_mock_otari_client())

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"custom_id": "req-1", "body": {"model": "gpt-4", "messages": []}}) + "\n")
        tmp_path = f.name

    try:
        with pytest.raises(InvalidRequestError, match="supports only /v1/chat/completions"):
            await provider._acreate_batch(input_file_path=tmp_path, endpoint="/v1/responses")
    finally:
        import os

        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_otari_retrieve_batch_requires_provider_name() -> None:
    provider = _build_provider(_mock_otari_client())

    with pytest.raises(InvalidRequestError, match="provider_name is required"):
        await provider._aretrieve_batch("batch-123")


@pytest.mark.asyncio
async def test_otari_list_batches_uses_sdk_client() -> None:
    mocked_client = _mock_otari_client()
    mocked_client.list_batches.return_value = [
        {
            "id": "batch-1",
            "object": "batch",
            "endpoint": "/v1/chat/completions",
            "input_file_id": "input-file-1",
            "status": "completed",
            "created_at": 1700000000,
            "completion_window": "24h",
            "request_counts": {"total": 1, "completed": 1, "failed": 0},
        }
    ]
    provider = _build_provider(mocked_client)

    result = await provider._alist_batches(provider_name="openai")

    assert len(result) == 1
    assert result[0].id == "batch-1"
    mocked_client.list_batches.assert_awaited_once_with(provider="openai", options=None)


@pytest.mark.asyncio
async def test_otari_retrieve_batch_results_maps_response() -> None:
    mocked_client = _mock_otari_client()
    mocked_client.retrieve_batch_results.return_value = {
        "results": [
            {
                "custom_id": "req-1",
                "result": {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "gpt-4",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                },
                "error": None,
            }
        ]
    }
    provider = _build_provider(mocked_client)

    result = await provider._aretrieve_batch_results("batch-1", provider_name="openai")

    assert isinstance(result, BatchResult)
    assert len(result.results) == 1
    assert result.results[0].custom_id == "req-1"
    assert result.results[0].result is not None
