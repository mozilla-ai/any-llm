import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from together.types import BatchJob

from any_llm.exceptions import BatchNotCompleteError, ProviderError, UnsupportedParameterError
from any_llm.providers.together.together import TogetherProvider
from any_llm.providers.together.utils import _convert_batch_job_to_openai

CREATED_AT = datetime(2026, 8, 12, 12, 0, 0, tzinfo=UTC)
CREATED_AT_EPOCH = int(CREATED_AT.timestamp())

# Together's deadline can land a fraction of a second short of a whole hour.
ALMOST_24H = timedelta(hours=24) - timedelta(microseconds=2)


def make_batch_job(**overrides: Any) -> BatchJob:
    """Build a Together BatchJob the lenient way the SDK parses API responses.

    Strict construction rejects statuses missing from the SDK's Literal, and the API
    returns at least one of those, so tests go through ``construct`` like the SDK does.
    """
    fields: dict[str, Any] = {
        "id": "batch-123",
        "status": "IN_PROGRESS",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-in",
        "created_at": CREATED_AT,
        "job_deadline": CREATED_AT + ALMOST_24H,
        "model_id": "openai/gpt-oss-20b",
        "output_file_id": None,
        "error_file_id": None,
        "completed_at": None,
    }
    fields.update(overrides)
    return BatchJob.construct(**fields)


def make_provider() -> tuple[TogetherProvider, Mock]:
    with patch("any_llm.providers.together.together.together.AsyncTogether") as mock_together:
        mock_client = Mock()
        mock_together.return_value = mock_client
        provider = TogetherProvider(api_key="test-api-key")
    return provider, mock_client


def make_output_file(*entries: dict[str, Any]) -> Mock:
    content = Mock()
    content.read = AsyncMock(return_value=("\n".join(json.dumps(entry) for entry in entries) + "\n").encode())
    return content


def test_convert_batch_job_maps_fields_to_openai_batch() -> None:
    result = _convert_batch_job_to_openai(
        make_batch_job(status="COMPLETED", output_file_id="file-out", completed_at=CREATED_AT + timedelta(hours=1))
    )

    assert result.id == "batch-123"
    assert result.object == "batch"
    assert result.status == "completed"
    assert result.endpoint == "/v1/chat/completions"
    assert result.input_file_id == "file-in"
    assert result.output_file_id == "file-out"
    assert result.model == "openai/gpt-oss-20b"
    assert result.created_at == CREATED_AT_EPOCH
    assert result.completed_at == CREATED_AT_EPOCH + 3600
    assert result.expires_at == int((CREATED_AT + ALMOST_24H).timestamp())


@pytest.mark.parametrize(
    ("deadline", "expected"),
    [
        (CREATED_AT + ALMOST_24H, "24h"),
        (CREATED_AT + timedelta(hours=48), "48h"),
        (CREATED_AT - timedelta(hours=1), "24h"),
        (None, "24h"),
    ],
    ids=["rounds-up-to-whole-hour", "derives-non-default", "deadline-before-creation", "no-deadline"],
)
def test_convert_batch_job_completion_window(deadline: datetime | None, expected: str) -> None:
    """Together never echoes the requested window, so it is derived from the deadline."""
    assert _convert_batch_job_to_openai(make_batch_job(job_deadline=deadline)).completion_window == expected


@pytest.mark.parametrize(
    ("together_status", "expected"),
    [
        ("VALIDATING", "validating"),
        ("IN_PROGRESS", "in_progress"),
        ("COMPLETED", "completed"),
        ("FAILED", "failed"),
        ("EXPIRED", "expired"),
        ("CANCELING", "cancelling"),
        ("CANCELLING", "cancelling"),
        ("CANCELLED", "cancelled"),
        ("SOMETHING_NEW", "in_progress"),
    ],
)
def test_convert_batch_job_status_mapping(together_status: str, expected: str) -> None:
    assert _convert_batch_job_to_openai(make_batch_job(status=together_status)).status == expected


def test_convert_batch_job_missing_fields_use_placeholders() -> None:
    result = _convert_batch_job_to_openai(make_batch_job(id=None, endpoint=None, input_file_id=None, created_at=None))

    assert result.id == ""
    assert result.endpoint == ""
    assert result.input_file_id == ""
    assert result.created_at == 0


@pytest.mark.asyncio
async def test_acreate_batch_uploads_file_then_creates_job(tmp_path: Path) -> None:
    provider, mock_client = make_provider()
    input_file = tmp_path / "batch.jsonl"
    input_file.write_text('{"custom_id": "a"}\n')

    mock_client.files.upload = AsyncMock(return_value=Mock(id="file-in"))
    mock_client.batches.create = AsyncMock(return_value=Mock(job=make_batch_job(), warning=None))

    result = await provider._acreate_batch(
        input_file_path=str(input_file),
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    upload_kwargs = mock_client.files.upload.call_args[1]
    assert upload_kwargs["file"] == input_file
    assert upload_kwargs["purpose"] == "batch-api"
    assert upload_kwargs["check"] is False

    create_kwargs = mock_client.batches.create.call_args[1]
    assert create_kwargs["input_file_id"] == "file-in"
    assert create_kwargs["endpoint"] == "/v1/chat/completions"
    assert create_kwargs["completion_window"] == "24h"

    assert result.id == "batch-123"
    assert result.status == "in_progress"


@pytest.mark.asyncio
async def test_acreate_batch_warns_about_metadata_and_together_warning(tmp_path: Path) -> None:
    provider, mock_client = make_provider()
    input_file = tmp_path / "batch.jsonl"
    input_file.write_text('{"custom_id": "a"}\n')

    mock_client.files.upload = AsyncMock(return_value=Mock(id="file-in"))
    mock_client.batches.create = AsyncMock(
        return_value=Mock(job=make_batch_job(), warning="model was overridden"),
    )

    with patch("any_llm.providers.together.together.logger") as mock_logger:
        result = await provider._acreate_batch(
            input_file_path=str(input_file),
            endpoint="/v1/chat/completions",
            metadata={"description": "ignored"},
        )

    warnings = " ".join(str(call) for call in mock_logger.warning.call_args_list)
    assert "metadata" in warnings
    assert "model was overridden" in warnings
    assert "metadata" not in mock_client.batches.create.call_args[1]
    assert result.id == "batch-123"


@pytest.mark.asyncio
async def test_acreate_batch_raises_when_no_job_returned(tmp_path: Path) -> None:
    provider, mock_client = make_provider()
    input_file = tmp_path / "batch.jsonl"
    input_file.write_text('{"custom_id": "a"}\n')

    mock_client.files.upload = AsyncMock(return_value=Mock(id="file-in"))
    mock_client.batches.create = AsyncMock(return_value=Mock(job=None, warning="quota exceeded"))

    with pytest.raises(ProviderError, match="quota exceeded"):
        await provider._acreate_batch(input_file_path=str(input_file), endpoint="/v1/chat/completions")


@pytest.mark.asyncio
async def test_aretrieve_batch_converts_job() -> None:
    provider, mock_client = make_provider()
    mock_client.batches.retrieve = AsyncMock(return_value=make_batch_job(status="VALIDATING"))

    result = await provider._aretrieve_batch("batch-123")

    assert mock_client.batches.retrieve.call_args[0][0] == "batch-123"
    assert result.status == "validating"


@pytest.mark.asyncio
async def test_acancel_batch_maps_canceling_to_cancelling() -> None:
    """Together answers a cancel with 'CANCELING', which OpenAI spells 'cancelling'."""
    provider, mock_client = make_provider()
    mock_client.batches.cancel = AsyncMock(return_value=make_batch_job(status="CANCELING"))

    result = await provider._acancel_batch("batch-123")

    assert mock_client.batches.cancel.call_args[0][0] == "batch-123"
    assert result.status == "cancelling"


@pytest.mark.asyncio
async def test_alist_batches_converts_jobs_and_applies_limit_client_side() -> None:
    provider, mock_client = make_provider()
    mock_client.batches.list = AsyncMock(return_value=[make_batch_job(id=f"batch-{i}") for i in range(5)])

    assert [batch.id for batch in await provider._alist_batches()] == [f"batch-{i}" for i in range(5)]
    assert [batch.id for batch in await provider._alist_batches(limit=2)] == ["batch-0", "batch-1"]
    assert "limit" not in mock_client.batches.list.call_args[1]


@pytest.mark.asyncio
async def test_alist_batches_handles_null_response() -> None:
    """Together answers an empty batch list with a literal null body."""
    provider, mock_client = make_provider()
    mock_client.batches.list = AsyncMock(return_value=None)

    assert await provider._alist_batches() == []


@pytest.mark.asyncio
async def test_alist_batches_rejects_after() -> None:
    provider, mock_client = make_provider()
    mock_client.batches.list = AsyncMock(return_value=[])

    with pytest.raises(UnsupportedParameterError, match="after"):
        await provider._alist_batches(after="batch-1")

    mock_client.batches.list.assert_not_called()


@pytest.mark.asyncio
async def test_aretrieve_batch_results_parses_successes_and_errors() -> None:
    provider, mock_client = make_provider()
    mock_client.batches.retrieve = AsyncMock(
        return_value=make_batch_job(status="COMPLETED", output_file_id="file-out"),
    )
    mock_client.files.content = AsyncMock(
        return_value=make_output_file(
            {
                "custom_id": "req-1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "cmpl-1",
                        "object": "chat.completion",
                        "created": 1700000000,
                        "model": "openai/gpt-oss-20b",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": "Paris"},
                            }
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
                    },
                },
            },
            {"custom_id": "req-2", "error": {"code": "rate_limit", "message": "slow down"}},
            {"custom_id": "req-3"},
        )
    )

    result = await provider._aretrieve_batch_results("batch-123")

    assert mock_client.files.content.call_args[0][0] == "file-out"
    assert [item.custom_id for item in result.results] == ["req-1", "req-2", "req-3"]

    completed, failed, malformed = result.results

    assert completed.result is not None
    assert completed.result.choices[0].message.content == "Paris"
    assert completed.result.model == "openai/gpt-oss-20b"
    assert completed.error is None

    assert failed.result is None
    assert failed.error is not None
    assert failed.error.code == "rate_limit"
    assert failed.error.message == "slow down"

    assert malformed.error is not None
    assert malformed.error.code == "unknown"


@pytest.mark.asyncio
async def test_aretrieve_batch_results_raises_when_not_complete() -> None:
    provider, mock_client = make_provider()
    mock_client.batches.retrieve = AsyncMock(return_value=make_batch_job(status="IN_PROGRESS"))

    with pytest.raises(BatchNotCompleteError) as exc_info:
        await provider._aretrieve_batch_results("batch-123")

    assert exc_info.value.batch_id == "batch-123"
    assert exc_info.value.batch_status == "in_progress"


@pytest.mark.asyncio
async def test_aretrieve_batch_results_without_output_file_is_empty() -> None:
    provider, mock_client = make_provider()
    mock_client.batches.retrieve = AsyncMock(return_value=make_batch_job(status="COMPLETED", output_file_id=None))
    mock_client.files.content = AsyncMock()

    result = await provider._aretrieve_batch_results("batch-123")

    assert result.results == []
    mock_client.files.content.assert_not_called()
