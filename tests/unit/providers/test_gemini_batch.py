import json
import tempfile
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from any_llm.exceptions import BatchNotCompleteError


def _make_mock_batch_job(
    *,
    name: str = "batches/batch-123",
    state_value: str = "JOB_STATE_SUCCEEDED",
    model: str = "gemini-2.5-flash",
    display_name: str | None = "test-batch",
    successful_count: int = 5,
    failed_count: int = 0,
    incomplete_count: int = 0,
    create_time: datetime | None = None,
    gcs_output_directory: str | None = "gs://output-bucket/results/",
    gcs_uri: list[str] | None = None,
    file_name: str | None = None,
) -> Mock:
    """Create a mock Google BatchJob object."""
    if create_time is None:
        create_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

    mock = Mock()
    mock.name = name
    mock.model = model
    mock.display_name = display_name

    mock.state = Mock()
    mock.state.value = state_value

    mock.completion_stats = Mock()
    mock.completion_stats.successful_count = successful_count
    mock.completion_stats.failed_count = failed_count
    mock.completion_stats.incomplete_count = incomplete_count

    mock.create_time = create_time

    mock.output_info = Mock()
    mock.output_info.gcs_output_directory = gcs_output_directory
    mock.output_info.bigquery_output_table = None

    mock.src = Mock()
    mock.src.gcs_uri = gcs_uri
    mock.src.file_name = file_name
    mock.src.inlined_requests = None

    return mock


def test_status_mapping_succeeded() -> None:
    """Test conversion of SUCCEEDED state."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(state_value="JOB_STATE_SUCCEEDED")
    result = _convert_google_batch_job_to_openai_batch(job)
    assert result.status == "completed"


def test_status_mapping_running() -> None:
    """Test conversion of RUNNING state."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(state_value="JOB_STATE_RUNNING")
    result = _convert_google_batch_job_to_openai_batch(job)
    assert result.status == "in_progress"


def test_status_mapping_queued() -> None:
    """Test conversion of QUEUED state."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(state_value="JOB_STATE_QUEUED")
    result = _convert_google_batch_job_to_openai_batch(job)
    assert result.status == "validating"


def test_status_mapping_pending() -> None:
    """Test conversion of PENDING state."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(state_value="JOB_STATE_PENDING")
    result = _convert_google_batch_job_to_openai_batch(job)
    assert result.status == "validating"


def test_status_mapping_failed() -> None:
    """Test conversion of FAILED state."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(state_value="JOB_STATE_FAILED")
    result = _convert_google_batch_job_to_openai_batch(job)
    assert result.status == "failed"


def test_status_mapping_cancelling() -> None:
    """Test conversion of CANCELLING state."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(state_value="JOB_STATE_CANCELLING")
    result = _convert_google_batch_job_to_openai_batch(job)
    assert result.status == "cancelling"


def test_status_mapping_cancelled() -> None:
    """Test conversion of CANCELLED state."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(state_value="JOB_STATE_CANCELLED")
    result = _convert_google_batch_job_to_openai_batch(job)
    assert result.status == "cancelled"


def test_status_mapping_expired() -> None:
    """Test conversion of EXPIRED state."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(state_value="JOB_STATE_EXPIRED")
    result = _convert_google_batch_job_to_openai_batch(job)
    assert result.status == "expired"


def test_status_mapping_paused() -> None:
    """Test conversion of PAUSED state."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(state_value="JOB_STATE_PAUSED")
    result = _convert_google_batch_job_to_openai_batch(job)
    assert result.status == "in_progress"


def test_status_mapping_partially_succeeded() -> None:
    """Test conversion of PARTIALLY_SUCCEEDED state."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(state_value="JOB_STATE_PARTIALLY_SUCCEEDED", successful_count=3, failed_count=2)
    result = _convert_google_batch_job_to_openai_batch(job)
    assert result.status == "completed"


def test_status_mapping_unknown_defaults_to_in_progress() -> None:
    """Test that unknown state logs a warning and defaults to in_progress."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(state_value="JOB_STATE_SOME_NEW_STATE")
    with patch("any_llm.providers.gemini.utils.logger") as mock_logger:
        result = _convert_google_batch_job_to_openai_batch(job)

    assert result.status == "in_progress"
    mock_logger.warning.assert_called_once()
    assert "Unknown Google batch state" in mock_logger.warning.call_args[0][0]


def test_convert_job_request_counts() -> None:
    """Test that completion stats map to BatchRequestCounts correctly."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(successful_count=7, failed_count=2, incomplete_count=1)
    result = _convert_google_batch_job_to_openai_batch(job)

    assert result.request_counts is not None
    assert result.request_counts.total == 10
    assert result.request_counts.completed == 7
    assert result.request_counts.failed == 2


def test_convert_job_id_is_name() -> None:
    """Test that the batch ID is set to the Google batch job name."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(name="batches/my-batch-456")
    result = _convert_google_batch_job_to_openai_batch(job)
    assert result.id == "batches/my-batch-456"


def test_convert_job_metadata_includes_display_name() -> None:
    """Test that display_name is included in metadata."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(display_name="my-custom-batch")
    result = _convert_google_batch_job_to_openai_batch(job)

    assert result.metadata is not None
    assert result.metadata["displayName"] == "my-custom-batch"


def test_convert_job_no_display_name_no_metadata() -> None:
    """Test that missing display_name results in no metadata."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(display_name=None)
    result = _convert_google_batch_job_to_openai_batch(job)
    assert result.metadata is None


def test_convert_job_no_create_time_defaults_to_zero() -> None:
    """Test that missing create_time defaults created_at to 0."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job()
    job.create_time = None
    result = _convert_google_batch_job_to_openai_batch(job)
    assert result.created_at == 0


def test_convert_job_output_file_id_from_gcs() -> None:
    """Test that GCS output directory is mapped to output_file_id."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job(gcs_output_directory="gs://my-bucket/output/")
    result = _convert_google_batch_job_to_openai_batch(job)
    assert result.output_file_id == "gs://my-bucket/output/"


def test_convert_job_no_completion_stats() -> None:
    """Test conversion when completion_stats is None."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_job_to_openai_batch

    job = _make_mock_batch_job()
    job.completion_stats = None
    result = _convert_google_batch_job_to_openai_batch(job)

    assert result.request_counts is not None
    assert result.request_counts.total == 0
    assert result.request_counts.completed == 0
    assert result.request_counts.failed == 0


def test_convert_openai_request_to_inlined_request_basic() -> None:
    """Test converting a basic OpenAI request to InlinedRequest."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_openai_request_to_inlined_request

    entry = {
        "custom_id": "req-1",
        "body": {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        },
    }

    result = _convert_openai_request_to_inlined_request(entry)
    assert result.model == "gemini-2.5-flash"
    assert result.contents is not None
    assert result.metadata == {"custom_id": "req-1"}
    assert result.config is not None
    assert result.config.max_output_tokens == 100


def test_convert_openai_request_with_system_message() -> None:
    """Test converting a request with system instructions."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_openai_request_to_inlined_request

    entry = {
        "custom_id": "req-2",
        "body": {
            "model": "gemini-2.5-flash",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ],
        },
    }

    result = _convert_openai_request_to_inlined_request(entry)
    assert result.config is not None
    assert result.config.system_instruction == "You are helpful"


def test_convert_openai_request_with_optional_params() -> None:
    """Test converting a request with temperature, top_p, and stop."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_openai_request_to_inlined_request

    entry = {
        "custom_id": "req-3",
        "body": {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "stop": ["END"],
        },
    }

    result = _convert_openai_request_to_inlined_request(entry)
    assert result.config is not None
    assert result.config.temperature == 0.5
    assert result.config.top_p == 0.9
    assert result.config.stop_sequences == ["END"]


def test_convert_openai_request_no_config_when_empty() -> None:
    """Test that config is None when no optional params are provided."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_openai_request_to_inlined_request

    entry = {
        "custom_id": "req-4",
        "body": {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    }

    result = _convert_openai_request_to_inlined_request(entry)
    assert result.config is None


def test_convert_batch_output_success() -> None:
    """Test parsing batch output JSONL with successful records."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_output_to_result

    output_line = json.dumps({
        "request": {"metadata": {"custom_id": "req-1"}},
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello world!"}],
                        "role": "model",
                    },
                    "finish_reason": "STOP",
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "total_token_count": 15,
            },
        },
        "model": "gemini-2.5-flash",
    })

    result = _convert_google_batch_output_to_result([output_line])
    assert len(result.results) == 1
    assert result.results[0].custom_id == "req-1"
    assert result.results[0].result is not None
    assert result.results[0].error is None


def test_convert_batch_output_error() -> None:
    """Test parsing batch output JSONL with errored records."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_output_to_result

    output_line = json.dumps({
        "request": {"metadata": {"custom_id": "req-2"}},
        "error": {
            "code": 400,
            "message": "Invalid request",
        },
    })

    result = _convert_google_batch_output_to_result([output_line])
    assert len(result.results) == 1
    assert result.results[0].custom_id == "req-2"
    assert result.results[0].result is None
    assert result.results[0].error is not None
    assert result.results[0].error.code == "400"
    assert result.results[0].error.message == "Invalid request"


def test_convert_batch_output_empty_lines_skipped() -> None:
    """Test that empty lines in the output are skipped."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_output_to_result

    output_line = json.dumps({
        "request": {"metadata": {"custom_id": "req-1"}},
        "response": {
            "candidates": [{"content": {"parts": [{"text": "Hi"}], "role": "model"}, "finish_reason": "STOP"}],
            "usage_metadata": {"prompt_token_count": 5, "candidates_token_count": 2, "total_token_count": 7},
        },
        "model": "gemini-2.5-flash",
    })

    result = _convert_google_batch_output_to_result(["", output_line, "", "  "])
    assert len(result.results) == 1


def test_convert_batch_output_missing_fields() -> None:
    """Test that a record with neither response nor error gets an error entry."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_output_to_result

    output_line = json.dumps({"request": {"metadata": {"custom_id": "req-1"}}})
    result = _convert_google_batch_output_to_result([output_line])
    assert len(result.results) == 1
    assert result.results[0].error is not None
    assert result.results[0].error.code == "unknown"


def test_convert_batch_output_invalid_json_skipped() -> None:
    """Test that invalid JSON lines are skipped."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.utils import _convert_google_batch_output_to_result

    result = _convert_google_batch_output_to_result(["not valid json", "also {bad"])
    assert len(result.results) == 0


class _MockAsyncPager:
    """Mock async pager that wraps a list of items for async iteration."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __aiter__(self) -> "_MockAsyncPager":
        self._index = 0
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


def _create_provider_with_mock_client() -> Any:
    """Create a GoogleProvider subclass with a mocked google.genai client."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.gemini import GeminiProvider

    with patch("any_llm.providers.gemini.gemini.genai"):
        provider = GeminiProvider(api_key="test-key")

    mock_client = MagicMock()
    mock_aio = MagicMock()
    mock_client.aio = mock_aio
    mock_aio.batches = MagicMock()
    provider.client = mock_client

    return provider, mock_client


@pytest.mark.asyncio
async def test_acreate_batch() -> None:
    """Test creating a batch job with the Google provider."""
    pytest.importorskip("google.genai")

    provider, mock_client = _create_provider_with_mock_client()

    mock_result = _make_mock_batch_job(state_value="JOB_STATE_QUEUED")
    mock_client.aio.batches.create = AsyncMock(return_value=mock_result)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(
            json.dumps({
                "custom_id": "req-1",
                "body": {
                    "model": "gemini-2.5-flash",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            })
            + "\n"
        )
        tmp_path = f.name

    try:
        result = await provider._acreate_batch(
            input_file_path=tmp_path,
            endpoint="/v1/chat/completions",
        )

        assert result.id == "batches/batch-123"
        assert result.status == "validating"
        mock_client.aio.batches.create.assert_called_once()

        call_kwargs = mock_client.aio.batches.create.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert len(call_kwargs["src"]) == 1
    finally:
        import os

        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_acreate_batch_with_display_name_and_dest() -> None:
    """Test creating a batch with display_name and dest kwargs."""
    pytest.importorskip("google.genai")
    from google.genai import types

    provider, mock_client = _create_provider_with_mock_client()

    mock_result = _make_mock_batch_job()
    mock_client.aio.batches.create = AsyncMock(return_value=mock_result)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(
            json.dumps({
                "custom_id": "req-1",
                "body": {
                    "model": "gemini-2.5-flash",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            })
            + "\n"
        )
        tmp_path = f.name

    try:
        await provider._acreate_batch(
            input_file_path=tmp_path,
            endpoint="/v1/chat/completions",
            display_name="my-batch",
            dest="gs://my-bucket/output/",
        )

        call_kwargs = mock_client.aio.batches.create.call_args[1]
        config = call_kwargs["config"]
        assert isinstance(config, types.CreateBatchJobConfig)
        assert config.display_name == "my-batch"
        assert config.dest == "gs://my-bucket/output/"
    finally:
        import os

        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_acreate_batch_empty_lines_skipped() -> None:
    """Test that empty lines in JSONL are skipped during batch creation."""
    pytest.importorskip("google.genai")

    provider, mock_client = _create_provider_with_mock_client()

    mock_result = _make_mock_batch_job()
    mock_client.aio.batches.create = AsyncMock(return_value=mock_result)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n")
        f.write(
            json.dumps({
                "custom_id": "req-1",
                "body": {"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "Hi"}]},
            })
            + "\n"
        )
        f.write("\n")
        tmp_path = f.name

    try:
        await provider._acreate_batch(input_file_path=tmp_path, endpoint="/v1/chat/completions")

        call_kwargs = mock_client.aio.batches.create.call_args[1]
        assert len(call_kwargs["src"]) == 1
    finally:
        import os

        os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_aretrieve_batch() -> None:
    """Test retrieving a batch job."""
    pytest.importorskip("google.genai")

    provider, mock_client = _create_provider_with_mock_client()

    mock_result = _make_mock_batch_job(name="batches/get-123", state_value="JOB_STATE_RUNNING")
    mock_client.aio.batches.get = AsyncMock(return_value=mock_result)

    result = await provider._aretrieve_batch("batches/get-123")

    assert result.id == "batches/get-123"
    assert result.status == "in_progress"
    mock_client.aio.batches.get.assert_called_once_with(name="batches/get-123")


@pytest.mark.asyncio
async def test_acancel_batch() -> None:
    """Test cancelling a batch job."""
    pytest.importorskip("google.genai")

    provider, mock_client = _create_provider_with_mock_client()

    mock_client.aio.batches.cancel = AsyncMock(return_value=None)
    mock_result = _make_mock_batch_job(name="batches/cancel-123", state_value="JOB_STATE_CANCELLING")
    mock_client.aio.batches.get = AsyncMock(return_value=mock_result)

    result = await provider._acancel_batch("batches/cancel-123")

    assert result.id == "batches/cancel-123"
    assert result.status == "cancelling"
    mock_client.aio.batches.cancel.assert_called_once_with(name="batches/cancel-123")


@pytest.mark.asyncio
async def test_alist_batches() -> None:
    """Test listing batch jobs."""
    pytest.importorskip("google.genai")

    provider, mock_client = _create_provider_with_mock_client()

    mock_jobs = [
        _make_mock_batch_job(name="batches/job1", state_value="JOB_STATE_SUCCEEDED"),
        _make_mock_batch_job(name="batches/job2", state_value="JOB_STATE_RUNNING"),
    ]
    mock_client.aio.batches.list = AsyncMock(return_value=_MockAsyncPager(mock_jobs))

    result = await provider._alist_batches()

    assert len(result) == 2
    assert result[0].id == "batches/job1"
    assert result[0].status == "completed"
    assert result[1].id == "batches/job2"
    assert result[1].status == "in_progress"


@pytest.mark.asyncio
async def test_alist_batches_with_pagination() -> None:
    """Test listing batch jobs with limit and after parameters."""
    pytest.importorskip("google.genai")
    from google.genai import types

    provider, mock_client = _create_provider_with_mock_client()

    mock_client.aio.batches.list = AsyncMock(return_value=_MockAsyncPager([]))

    await provider._alist_batches(limit=5, after="next-page-token")

    call_kwargs = mock_client.aio.batches.list.call_args[1]
    config = call_kwargs["config"]
    assert isinstance(config, types.ListBatchJobsConfig)
    assert config.page_size == 5
    assert config.page_token == "next-page-token"


@pytest.mark.asyncio
async def test_aretrieve_batch_results_not_completed() -> None:
    """Test that BatchNotCompleteError is raised for incomplete batch."""
    pytest.importorskip("google.genai")

    provider, mock_client = _create_provider_with_mock_client()

    mock_job = _make_mock_batch_job(state_value="JOB_STATE_RUNNING")
    mock_client.aio.batches.get = AsyncMock(return_value=mock_job)

    with pytest.raises(BatchNotCompleteError) as exc_info:
        await provider._aretrieve_batch_results("batches/pending-123")

    assert exc_info.value.batch_id == "batches/pending-123"
    assert exc_info.value.batch_status == "in_progress"


@pytest.mark.asyncio
async def test_aretrieve_batch_results_no_output_directory() -> None:
    """Test that ValueError is raised when batch has no output directory."""
    pytest.importorskip("google.genai")

    provider, mock_client = _create_provider_with_mock_client()

    mock_job = _make_mock_batch_job(state_value="JOB_STATE_SUCCEEDED", gcs_output_directory=None)
    mock_client.aio.batches.get = AsyncMock(return_value=mock_job)

    with pytest.raises(ValueError, match="no GCS output directory"):
        await provider._aretrieve_batch_results("batches/no-output-123")


@pytest.mark.asyncio
async def test_aretrieve_batch_results_completed() -> None:
    """Test retrieving results from a completed batch job."""
    pytest.importorskip("google.genai")

    provider, mock_client = _create_provider_with_mock_client()

    mock_job = _make_mock_batch_job(
        state_value="JOB_STATE_SUCCEEDED",
        gcs_output_directory="gs://output-bucket/results/",
    )
    mock_client.aio.batches.get = AsyncMock(return_value=mock_job)

    output_line = json.dumps({
        "request": {"metadata": {"custom_id": "req-1"}},
        "response": {
            "candidates": [
                {"content": {"parts": [{"text": "Hello!"}], "role": "model"}, "finish_reason": "STOP"}
            ],
            "usage_metadata": {"prompt_token_count": 10, "candidates_token_count": 5, "total_token_count": 15},
        },
        "model": "gemini-2.5-flash",
    })

    with patch.object(provider, "_read_gcs_output", return_value=[output_line]):
        result = await provider._aretrieve_batch_results("batches/done-123")

    assert len(result.results) == 1
    assert result.results[0].custom_id == "req-1"
    assert result.results[0].result is not None
    assert result.results[0].error is None


@pytest.mark.asyncio
async def test_aretrieve_batch_results_partially_succeeded() -> None:
    """Test retrieving results from a partially succeeded batch job."""
    pytest.importorskip("google.genai")

    provider, mock_client = _create_provider_with_mock_client()

    mock_job = _make_mock_batch_job(
        state_value="JOB_STATE_PARTIALLY_SUCCEEDED",
        gcs_output_directory="gs://output-bucket/results/",
    )
    mock_client.aio.batches.get = AsyncMock(return_value=mock_job)

    success_line = json.dumps({
        "request": {"metadata": {"custom_id": "req-1"}},
        "response": {
            "candidates": [{"content": {"parts": [{"text": "OK"}], "role": "model"}, "finish_reason": "STOP"}],
            "usage_metadata": {"prompt_token_count": 5, "candidates_token_count": 2, "total_token_count": 7},
        },
        "model": "gemini-2.5-flash",
    })
    error_line = json.dumps({
        "request": {"metadata": {"custom_id": "req-2"}},
        "error": {"code": 500, "message": "Internal error"},
    })

    with patch.object(provider, "_read_gcs_output", return_value=[success_line, error_line]):
        result = await provider._aretrieve_batch_results("batches/partial-123")

    assert len(result.results) == 2
    assert result.results[0].result is not None
    assert result.results[1].error is not None
    assert result.results[1].error.code == "500"


def test_read_gcs_output_requires_google_cloud_storage() -> None:
    """Test that _read_gcs_output raises ImportError when google-cloud-storage is missing."""
    pytest.importorskip("google.genai")
    from any_llm.providers.gemini.base import GoogleProvider

    with patch.dict("sys.modules", {"google.cloud": None, "google.cloud.storage": None}):
        with pytest.raises(ImportError, match="google-cloud-storage"):
            GoogleProvider._read_gcs_output("gs://bucket/output/")


def test_read_gcs_output_invalid_uri() -> None:
    """Test that _read_gcs_output raises ValueError for non-GCS URIs."""
    pytest.importorskip("google.genai")
    pytest.importorskip("google.cloud.storage")
    from any_llm.providers.gemini.base import GoogleProvider

    with pytest.raises(ValueError, match="Expected a GCS URI"):
        GoogleProvider._read_gcs_output("s3://wrong-scheme/output/")
