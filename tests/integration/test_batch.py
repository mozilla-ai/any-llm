import json
import tempfile
from pathlib import Path
from typing import Any

import httpx
import pytest
from openai import APIConnectionError

from any_llm import AnyLLM, LLMProvider
from any_llm.api import acancel_batch, acreate_batch, alist_batches, aretrieve_batch
from any_llm.exceptions import MissingApiKeyError
from any_llm.types.batch import Batch
from tests.constants import EXPECTED_PROVIDERS, LOCAL_PROVIDERS


@pytest.mark.asyncio
async def test_create_and_retrieve_batch(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test creating and retrieving a batch job."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_BATCH:
            pytest.skip(f"{provider.value} does not support batch completions, skipping")

        model_id = provider_model_map[provider]

        batch_requests = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_id,
                    "messages": [{"role": "user", "content": "What is 2+2?"}],
                    "max_tokens": 100,
                },
            },
            {
                "custom_id": "request-2",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_id,
                    "messages": [{"role": "user", "content": "What is the capital of France?"}],
                    "max_tokens": 100,
                },
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_file:
            for request in batch_requests:
                tmp_file.write(json.dumps(request) + "\n")
            tmp_file_path = Path(tmp_file.name)

        try:
            batch = await llm.acreate_batch(
                input_file_path=str(tmp_file_path),
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": "Test batch from any-llm integration tests"},
            )

            assert isinstance(batch, Batch)
            assert batch.id is not None
            assert batch.status in [
                "validating",
                "in_progress",
                "finalizing",
                "completed",
                "failed",
                "expired",
                "cancelled",
            ]
            assert batch.input_file_id is not None
            assert batch.endpoint == "/v1/chat/completions"
            assert batch.completion_window == "24h"

            retrieved_batch = await llm.aretrieve_batch(batch.id)
            assert isinstance(retrieved_batch, Batch)
            assert retrieved_batch.id == batch.id
            assert retrieved_batch.input_file_id == batch.input_file_id

            cancelled_batch = await llm.acancel_batch(batch.id)
            assert isinstance(cancelled_batch, Batch)
            assert cancelled_batch.id == batch.id
            assert cancelled_batch.status in ["cancelling", "cancelled"]

        finally:
            tmp_file_path.unlink(missing_ok=True)

    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local model host is not set up, skipping")
        raise


@pytest.mark.asyncio
async def test_list_batches(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test listing batch jobs."""
    try:
        llm = AnyLLM.create(provider, **provider_client_config.get(provider, {}))
        if not llm.SUPPORTS_BATCH:
            pytest.skip(f"{provider.value} does not support batch completions, skipping")

        batches = await llm.alist_batches()
        assert isinstance(batches, list)

        limited_batches = await llm.alist_batches(limit=5)
        assert isinstance(limited_batches, list)
        assert len(limited_batches) <= 5

    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local model host is not set up, skipping")
        raise


@pytest.mark.asyncio
async def test_batch_with_api_functions(
    provider: LLMProvider,
    provider_model_map: dict[LLMProvider, str],
    provider_client_config: dict[LLMProvider, dict[str, Any]],
) -> None:
    """Test batch operations using the top-level API functions."""
    try:
        if provider not in [LLMProvider.OPENAI]:
            pytest.skip(f"{provider.value} does not support batch completions, skipping")

        model_id = provider_model_map[provider]

        batch_requests = [
            {
                "custom_id": "api-test-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_id,
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_tokens": 50,
                },
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_file:
            for request in batch_requests:
                tmp_file.write(json.dumps(request) + "\n")
            tmp_file_path = Path(tmp_file.name)

        try:
            batch = await acreate_batch(
                provider=provider,
                input_file_path=str(tmp_file_path),
                endpoint="/v1/chat/completions",
                **provider_client_config.get(provider, {}),
            )
            assert isinstance(batch, Batch)
            assert batch.id is not None

            retrieved = await aretrieve_batch(
                provider=provider,
                batch_id=batch.id,
                **provider_client_config.get(provider, {}),
            )
            assert isinstance(retrieved, Batch)
            assert retrieved.id == batch.id

            batches = await alist_batches(
                provider=provider,
                limit=10,
                **provider_client_config.get(provider, {}),
            )
            assert isinstance(batches, list)

            cancelled = await acancel_batch(
                provider=provider,
                batch_id=batch.id,
                **provider_client_config.get(provider, {}),
            )
            assert isinstance(cancelled, Batch)
            assert cancelled.id == batch.id

        finally:
            tmp_file_path.unlink(missing_ok=True)

    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in LOCAL_PROVIDERS and provider not in EXPECTED_PROVIDERS:
            pytest.skip("Local model host is not set up, skipping")
        raise
