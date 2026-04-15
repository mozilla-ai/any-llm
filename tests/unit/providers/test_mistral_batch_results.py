import json
from unittest.mock import AsyncMock, Mock, patch

import pytest


@pytest.mark.asyncio
async def test_aretrieve_batch_results_success() -> None:
    """Test retrieving batch results with successes and errors from Mistral."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider
    from any_llm.types.batch import BatchResult

    with patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral:
        provider = MistralProvider(api_key="test-api-key")

        mock_batch_job = Mock()
        mock_batch_job.id = "batch-results-123"
        mock_batch_job.input_files = ["file-abc"]
        mock_batch_job.endpoint = "/v1/chat/completions"
        mock_batch_job.status = Mock(value="SUCCESS")
        mock_batch_job.created_at = 1700000000
        mock_batch_job.total_requests = 2
        mock_batch_job.completed_requests = 2
        mock_batch_job.succeeded_requests = 1
        mock_batch_job.failed_requests = 1
        mock_batch_job.errors = []
        mock_batch_job.metadata = None
        mock_batch_job.output_file = "output-file-123"
        mock_batch_job.error_file = None
        mock_batch_job.started_at = 1700000100
        mock_batch_job.completed_at = 1700000200

        mocked_mistral.return_value.batch.jobs.get_async = AsyncMock(return_value=mock_batch_job)

        success_line = json.dumps(
            {
                "custom_id": "req-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "mistral-small-latest",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "Paris"},
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
                "error": {"code": "rate_limit", "message": "Rate limit exceeded"},
            }
        )

        file_content = f"{success_line}\n{error_line}".encode()
        mocked_mistral.return_value.files.download_async = AsyncMock(return_value=file_content)

        result = await provider._aretrieve_batch_results("batch-results-123")

        assert isinstance(result, BatchResult)
        assert len(result.results) == 2
        assert result.results[0].custom_id == "req-1"
        assert result.results[0].result is not None
        assert result.results[0].result.id == "chatcmpl-1"
        assert result.results[1].custom_id == "req-2"
        assert result.results[1].error is not None
        assert result.results[1].error.code == "rate_limit"


@pytest.mark.asyncio
async def test_aretrieve_batch_results_not_completed() -> None:
    """Test that BatchNotCompleteError is raised for non-completed Mistral batch."""
    pytest.importorskip("mistralai")
    from any_llm.exceptions import BatchNotCompleteError
    from any_llm.providers.mistral.mistral import MistralProvider

    with patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral:
        provider = MistralProvider(api_key="test-api-key")

        mock_batch_job = Mock()
        mock_batch_job.id = "batch-not-done"
        mock_batch_job.input_files = ["file-abc"]
        mock_batch_job.endpoint = "/v1/chat/completions"
        mock_batch_job.status = Mock(value="RUNNING")
        mock_batch_job.created_at = 1700000000
        mock_batch_job.total_requests = 10
        mock_batch_job.completed_requests = 3
        mock_batch_job.succeeded_requests = 3
        mock_batch_job.failed_requests = 0
        mock_batch_job.errors = []
        mock_batch_job.metadata = None
        mock_batch_job.output_file = None
        mock_batch_job.error_file = None
        mock_batch_job.started_at = 1700000100
        mock_batch_job.completed_at = None

        mocked_mistral.return_value.batch.jobs.get_async = AsyncMock(return_value=mock_batch_job)

        with pytest.raises(BatchNotCompleteError) as exc_info:
            await provider._aretrieve_batch_results("batch-not-done")

        assert exc_info.value.batch_id == "batch-not-done"


@pytest.mark.asyncio
async def test_aretrieve_batch_results_empty_output() -> None:
    """Test retrieving results when there is no output file."""
    pytest.importorskip("mistralai")
    from any_llm.providers.mistral.mistral import MistralProvider
    from any_llm.types.batch import BatchResult

    with patch("any_llm.providers.mistral.mistral.Mistral") as mocked_mistral:
        provider = MistralProvider(api_key="test-api-key")

        mock_batch_job = Mock()
        mock_batch_job.id = "batch-empty"
        mock_batch_job.input_files = ["file-abc"]
        mock_batch_job.endpoint = "/v1/chat/completions"
        mock_batch_job.status = Mock(value="SUCCESS")
        mock_batch_job.created_at = 1700000000
        mock_batch_job.total_requests = 0
        mock_batch_job.completed_requests = 0
        mock_batch_job.succeeded_requests = 0
        mock_batch_job.failed_requests = 0
        mock_batch_job.errors = []
        mock_batch_job.metadata = None
        mock_batch_job.output_file = None
        mock_batch_job.error_file = None
        mock_batch_job.started_at = 1700000100
        mock_batch_job.completed_at = 1700000200

        mocked_mistral.return_value.batch.jobs.get_async = AsyncMock(return_value=mock_batch_job)

        result = await provider._aretrieve_batch_results("batch-empty")

        assert isinstance(result, BatchResult)
        assert len(result.results) == 0
