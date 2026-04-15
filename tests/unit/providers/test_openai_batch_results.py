import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from any_llm.exceptions import BatchNotCompleteError
from any_llm.types.batch import BatchResult


@pytest.mark.asyncio
async def test_aretrieve_batch_results_not_supported() -> None:
    """Test that _aretrieve_batch_results raises NotImplementedError when SUPPORTS_BATCH is False."""
    from any_llm.providers.openai.openai import OpenaiProvider

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = OpenaiProvider(api_key="test-key")
        provider.SUPPORTS_BATCH = False

        with pytest.raises(NotImplementedError, match="does not support batch"):
            await provider._aretrieve_batch_results("batch-123")


@pytest.mark.asyncio
async def test_aretrieve_batch_results_success() -> None:
    """Test retrieving batch results with mixed successes and failures."""
    from any_llm.providers.openai.openai import OpenaiProvider

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = OpenaiProvider(api_key="test-key")

        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-output-123"
        provider.client.batches.retrieve = AsyncMock(return_value=mock_batch)  # type: ignore[method-assign]

        success_line = json.dumps(
            {
                "custom_id": "req-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "4"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                    },
                },
            }
        )
        error_line = json.dumps(
            {
                "custom_id": "req-2",
                "error": {
                    "code": "invalid_request",
                    "message": "Invalid model",
                },
            }
        )

        mock_content = MagicMock()
        mock_content.text = f"{success_line}\n{error_line}"
        provider.client.files.content = AsyncMock(return_value=mock_content)  # type: ignore[method-assign]

        result = await provider._aretrieve_batch_results("batch-123")

        assert isinstance(result, BatchResult)
        assert len(result.results) == 2

        assert result.results[0].custom_id == "req-1"
        assert result.results[0].result is not None
        assert result.results[0].result.id == "chatcmpl-1"
        assert result.results[0].error is None

        assert result.results[1].custom_id == "req-2"
        assert result.results[1].result is None
        assert result.results[1].error is not None
        assert result.results[1].error.code == "invalid_request"
        assert result.results[1].error.message == "Invalid model"


@pytest.mark.asyncio
async def test_aretrieve_batch_results_not_completed() -> None:
    """Test that BatchNotCompleteError is raised for non-completed batch."""
    from any_llm.providers.openai.openai import OpenaiProvider

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = OpenaiProvider(api_key="test-key")

        mock_batch = MagicMock()
        mock_batch.status = "in_progress"
        provider.client.batches.retrieve = AsyncMock(return_value=mock_batch)  # type: ignore[method-assign]

        with pytest.raises(BatchNotCompleteError) as exc_info:
            await provider._aretrieve_batch_results("batch-456")

        assert exc_info.value.batch_id == "batch-456"
        assert exc_info.value.batch_status == "in_progress"


@pytest.mark.asyncio
async def test_aretrieve_batch_results_empty_output_file() -> None:
    """Test retrieving results when output file is empty (no output_file_id)."""
    from any_llm.providers.openai.openai import OpenaiProvider

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = OpenaiProvider(api_key="test-key")

        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = None
        provider.client.batches.retrieve = AsyncMock(return_value=mock_batch)  # type: ignore[method-assign]

        result = await provider._aretrieve_batch_results("batch-789")

        assert isinstance(result, BatchResult)
        assert len(result.results) == 0


@pytest.mark.asyncio
async def test_aretrieve_batch_results_skips_empty_lines() -> None:
    """Test that empty lines in the JSONL output are skipped."""
    from any_llm.providers.openai.openai import OpenaiProvider

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = OpenaiProvider(api_key="test-key")

        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-output-empty-lines"
        provider.client.batches.retrieve = AsyncMock(return_value=mock_batch)  # type: ignore[method-assign]

        line1 = json.dumps(
            {
                "custom_id": "req-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                    },
                },
            }
        )
        line2 = json.dumps(
            {
                "custom_id": "req-2",
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "chatcmpl-2",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "gpt-4",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                    },
                },
            }
        )
        mock_content = MagicMock()
        mock_content.text = f"{line1}\n  \n{line2}"
        provider.client.files.content = AsyncMock(return_value=mock_content)  # type: ignore[method-assign]

        result = await provider._aretrieve_batch_results("batch-empty-lines")

        assert len(result.results) == 2
        assert result.results[0].custom_id == "req-1"
        assert result.results[1].custom_id == "req-2"


@pytest.mark.asyncio
async def test_aretrieve_batch_results_unexpected_format() -> None:
    """Test that unexpected response format produces an error item."""
    from any_llm.providers.openai.openai import OpenaiProvider

    with patch("any_llm.providers.openai.base.AsyncOpenAI"):
        provider = OpenaiProvider(api_key="test-key")

        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-output-unexpected"
        provider.client.batches.retrieve = AsyncMock(return_value=mock_batch)  # type: ignore[method-assign]

        unexpected_line = json.dumps(
            {
                "custom_id": "req-3",
            }
        )
        mock_content = MagicMock()
        mock_content.text = unexpected_line
        provider.client.files.content = AsyncMock(return_value=mock_content)  # type: ignore[method-assign]

        result = await provider._aretrieve_batch_results("batch-unexpected")

        assert len(result.results) == 1
        assert result.results[0].custom_id == "req-3"
        assert result.results[0].error is not None
        assert result.results[0].error.code == "unknown"
