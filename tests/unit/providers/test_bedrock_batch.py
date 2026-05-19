import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from any_llm.exceptions import BatchNotCompleteError, InvalidRequestError


def _make_mock_job(
    *,
    job_arn: str = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/abc123",
    job_name: str = "test-batch-job",
    status: str = "Completed",
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0",
    total_records: int = 5,
    success_records: int = 5,
    error_records: int = 0,
    input_s3_uri: str = "s3://input-bucket/batch-input.jsonl",
    output_s3_uri: str = "s3://output-bucket/results/",
    submit_time: datetime | None = None,
) -> dict[str, Any]:
    """Create a mock Bedrock GetModelInvocationJob response."""
    if submit_time is None:
        submit_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

    return {
        "jobArn": job_arn,
        "jobName": job_name,
        "status": status,
        "modelId": model_id,
        "totalRecordCount": total_records,
        "successRecordCount": success_records,
        "errorRecordCount": error_records,
        "processedRecordCount": success_records + error_records,
        "submitTime": submit_time,
        "inputDataConfig": {"s3InputDataConfig": {"s3Uri": input_s3_uri}},
        "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": output_s3_uri}},
    }


def test_status_mapping_completed() -> None:
    """Test conversion of Completed status."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job(status="Completed")
    result = _convert_bedrock_job_to_openai_batch(job)
    assert result.status == "completed"


def test_status_mapping_in_progress() -> None:
    """Test conversion of InProgress status."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job(status="InProgress")
    result = _convert_bedrock_job_to_openai_batch(job)
    assert result.status == "in_progress"


def test_status_mapping_submitted() -> None:
    """Test conversion of Submitted status."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job(status="Submitted")
    result = _convert_bedrock_job_to_openai_batch(job)
    assert result.status == "validating"


def test_status_mapping_validating() -> None:
    """Test conversion of Validating status."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job(status="Validating")
    result = _convert_bedrock_job_to_openai_batch(job)
    assert result.status == "validating"


def test_status_mapping_scheduled() -> None:
    """Test conversion of Scheduled status."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job(status="Scheduled")
    result = _convert_bedrock_job_to_openai_batch(job)
    assert result.status == "validating"


def test_status_mapping_failed() -> None:
    """Test conversion of Failed status."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job(status="Failed")
    result = _convert_bedrock_job_to_openai_batch(job)
    assert result.status == "failed"


def test_status_mapping_stopping() -> None:
    """Test conversion of Stopping status."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job(status="Stopping")
    result = _convert_bedrock_job_to_openai_batch(job)
    assert result.status == "cancelling"


def test_status_mapping_stopped() -> None:
    """Test conversion of Stopped status."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job(status="Stopped")
    result = _convert_bedrock_job_to_openai_batch(job)
    assert result.status == "cancelled"


def test_status_mapping_expired() -> None:
    """Test conversion of Expired status."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job(status="Expired")
    result = _convert_bedrock_job_to_openai_batch(job)
    assert result.status == "expired"


def test_status_mapping_partially_completed() -> None:
    """Test conversion of PartiallyCompleted status."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job(status="PartiallyCompleted", success_records=3, error_records=2)
    result = _convert_bedrock_job_to_openai_batch(job)
    assert result.status == "completed"


def test_status_mapping_unknown_defaults_to_in_progress() -> None:
    """Test that unknown status logs a warning and defaults to in_progress."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job(status="SomeNewStatus")
    with patch("any_llm.providers.bedrock.utils.logger") as mock_logger:
        result = _convert_bedrock_job_to_openai_batch(job)

    assert result.status == "in_progress"
    mock_logger.warning.assert_called_once()
    assert "Unknown Bedrock batch status" in mock_logger.warning.call_args[0][0]


def test_convert_job_request_counts() -> None:
    """Test that record counts map to BatchRequestCounts correctly."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job(total_records=10, success_records=7, error_records=3)
    result = _convert_bedrock_job_to_openai_batch(job)

    assert result.request_counts is not None
    assert result.request_counts.total == 10
    assert result.request_counts.completed == 7
    assert result.request_counts.failed == 3


def test_convert_job_id_is_job_arn() -> None:
    """Test that the batch ID is set to the Bedrock job ARN."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    arn = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/xyz789"
    job = _make_mock_job(job_arn=arn)
    result = _convert_bedrock_job_to_openai_batch(job)

    assert result.id == arn


def test_convert_job_metadata_includes_job_name() -> None:
    """Test that job name is included in metadata."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job(job_name="my-custom-job")
    result = _convert_bedrock_job_to_openai_batch(job)

    assert result.metadata is not None
    assert result.metadata["jobName"] == "my-custom-job"


def test_convert_job_no_submit_time_defaults_to_zero() -> None:
    """Test that missing submit_time defaults created_at to 0."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job()
    job["submitTime"] = None
    result = _convert_bedrock_job_to_openai_batch(job)

    assert result.created_at == 0


def test_convert_job_input_output_file_ids() -> None:
    """Test that input/output S3 URIs are mapped to file IDs."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_job_to_openai_batch

    job = _make_mock_job(
        input_s3_uri="s3://my-bucket/input.jsonl",
        output_s3_uri="s3://my-bucket/output/",
    )
    result = _convert_bedrock_job_to_openai_batch(job)

    assert result.input_file_id == "s3://my-bucket/input.jsonl"
    assert result.output_file_id == "s3://my-bucket/output/"


def test_parse_s3_uri_valid() -> None:
    """Test parsing a valid S3 URI."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _parse_s3_uri

    bucket, key = _parse_s3_uri("s3://my-bucket/path/to/file.jsonl")
    assert bucket == "my-bucket"
    assert key == "path/to/file.jsonl"


def test_parse_s3_uri_single_level_key() -> None:
    """Test parsing an S3 URI with a single-level key."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _parse_s3_uri

    bucket, key = _parse_s3_uri("s3://bucket/file.jsonl")
    assert bucket == "bucket"
    assert key == "file.jsonl"


def test_parse_s3_uri_invalid_scheme() -> None:
    """Test that a non-S3 URI raises InvalidRequestError."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _parse_s3_uri

    with pytest.raises(InvalidRequestError, match="Expected an S3 URI"):
        _parse_s3_uri("https://example.com/file.jsonl")


def test_parse_s3_uri_no_key() -> None:
    """Test that an S3 URI without a key raises InvalidRequestError."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _parse_s3_uri

    with pytest.raises(InvalidRequestError, match="must include a key"):
        _parse_s3_uri("s3://bucket-only")


def test_parse_s3_uri_empty_key() -> None:
    """Test that an S3 URI with an empty key raises InvalidRequestError."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _parse_s3_uri

    with pytest.raises(InvalidRequestError, match="empty bucket or key"):
        _parse_s3_uri("s3://bucket/")


def test_convert_batch_output_success() -> None:
    """Test parsing batch output JSONL with successful records."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_batch_output_to_result

    output_line = json.dumps(
        {
            "recordId": "req-1",
            "modelOutput": {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Hello world!"}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 10, "outputTokens": 5},
            },
        }
    )

    result = _convert_bedrock_batch_output_to_result([output_line])
    assert len(result.results) == 1
    assert result.results[0].custom_id == "req-1"
    assert result.results[0].result is not None
    assert result.results[0].error is None


def test_convert_batch_output_error() -> None:
    """Test parsing batch output JSONL with errored records."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_batch_output_to_result

    output_line = json.dumps(
        {
            "recordId": "req-2",
            "error": {
                "errorCode": "ThrottlingException",
                "errorMessage": "Rate exceeded",
            },
        }
    )

    result = _convert_bedrock_batch_output_to_result([output_line])
    assert len(result.results) == 1
    assert result.results[0].custom_id == "req-2"
    assert result.results[0].result is None
    assert result.results[0].error is not None
    assert result.results[0].error.code == "ThrottlingException"
    assert result.results[0].error.message == "Rate exceeded"


def test_convert_batch_output_mixed() -> None:
    """Test parsing batch output with mixed success and error records."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_batch_output_to_result

    success_line = json.dumps(
        {
            "recordId": "req-1",
            "modelOutput": {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Response 1"}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 10, "outputTokens": 5},
            },
        }
    )
    error_line = json.dumps(
        {
            "recordId": "req-2",
            "error": {"errorCode": "ValidationException", "errorMessage": "Invalid input"},
        }
    )

    result = _convert_bedrock_batch_output_to_result([success_line, error_line])
    assert len(result.results) == 2
    assert result.results[0].result is not None
    assert result.results[1].error is not None


def test_convert_batch_output_empty_lines_skipped() -> None:
    """Test that empty lines in the output are skipped."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_batch_output_to_result

    output_line = json.dumps(
        {
            "recordId": "req-1",
            "modelOutput": {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Hello"}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 5, "outputTokens": 3},
            },
        }
    )

    result = _convert_bedrock_batch_output_to_result(["", output_line, "", "  "])
    assert len(result.results) == 1


def test_convert_batch_output_invalid_json_skipped() -> None:
    """Test that invalid JSON lines are skipped."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_batch_output_to_result

    result = _convert_bedrock_batch_output_to_result(["not valid json", "also {bad"])
    assert len(result.results) == 0


def test_convert_batch_output_missing_fields() -> None:
    """Test that a record with neither modelOutput nor error gets an error entry."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.utils import _convert_bedrock_batch_output_to_result

    output_line = json.dumps({"recordId": "req-1"})
    result = _convert_bedrock_batch_output_to_result([output_line])
    assert len(result.results) == 1
    assert result.results[0].error is not None
    assert result.results[0].error.code == "unknown"


def _create_provider_with_mock_clients() -> Any:
    """Create a BedrockProvider with mocked boto3 clients."""
    from any_llm.providers.bedrock.bedrock import BedrockProvider

    mock_runtime_client = MagicMock()
    provider = BedrockProvider(client=mock_runtime_client)

    mock_control_client = MagicMock()
    mock_s3_client = MagicMock()
    provider._get_bedrock_control_client = Mock(return_value=mock_control_client)  # type: ignore[method-assign]
    provider._get_s3_client = Mock(return_value=mock_s3_client)  # type: ignore[method-assign]

    return provider, mock_control_client, mock_s3_client


@pytest.mark.asyncio
async def test_acreate_batch() -> None:
    """Test creating a batch job with the Bedrock provider."""
    pytest.importorskip("boto3")

    provider, mock_control, _ = _create_provider_with_mock_clients()

    job_arn = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/new123"
    mock_control.create_model_invocation_job.return_value = {"jobArn": job_arn}
    mock_control.get_model_invocation_job.return_value = _make_mock_job(job_arn=job_arn, status="Submitted")

    result = await provider._acreate_batch(
        input_file_path="s3://input-bucket/batch.jsonl",
        endpoint="/v1/chat/completions",
        role_arn="arn:aws:iam::123456789012:role/BatchRole",
        output_s3_uri="s3://output-bucket/results/",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
    )

    assert result.id == job_arn
    assert result.status == "validating"
    mock_control.create_model_invocation_job.assert_called_once()
    call_kwargs = mock_control.create_model_invocation_job.call_args[1]
    assert call_kwargs["roleArn"] == "arn:aws:iam::123456789012:role/BatchRole"
    assert call_kwargs["modelId"] == "anthropic.claude-3-haiku-20240307-v1:0"
    assert call_kwargs["modelInvocationType"] == "Converse"
    assert call_kwargs["inputDataConfig"]["s3InputDataConfig"]["s3Uri"] == "s3://input-bucket/batch.jsonl"
    assert call_kwargs["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"] == "s3://output-bucket/results/"


@pytest.mark.asyncio
async def test_acreate_batch_missing_role_arn() -> None:
    """Test that missing role_arn raises InvalidRequestError."""
    pytest.importorskip("boto3")

    provider, _, _ = _create_provider_with_mock_clients()

    with pytest.raises(InvalidRequestError, match="role_arn"):
        await provider._acreate_batch(
            input_file_path="s3://bucket/input.jsonl",
            endpoint="/v1/chat/completions",
            output_s3_uri="s3://bucket/output/",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
        )


@pytest.mark.asyncio
async def test_acreate_batch_missing_output_s3_uri() -> None:
    """Test that missing output_s3_uri raises InvalidRequestError."""
    pytest.importorskip("boto3")

    provider, _, _ = _create_provider_with_mock_clients()

    with pytest.raises(InvalidRequestError, match="output_s3_uri"):
        await provider._acreate_batch(
            input_file_path="s3://bucket/input.jsonl",
            endpoint="/v1/chat/completions",
            role_arn="arn:aws:iam::123456789012:role/BatchRole",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
        )


@pytest.mark.asyncio
async def test_acreate_batch_missing_model_id() -> None:
    """Test that missing model_id raises InvalidRequestError."""
    pytest.importorskip("boto3")

    provider, _, _ = _create_provider_with_mock_clients()

    with pytest.raises(InvalidRequestError, match="model_id"):
        await provider._acreate_batch(
            input_file_path="s3://bucket/input.jsonl",
            endpoint="/v1/chat/completions",
            role_arn="arn:aws:iam::123456789012:role/BatchRole",
            output_s3_uri="s3://bucket/output/",
        )


@pytest.mark.asyncio
async def test_acreate_batch_with_metadata() -> None:
    """Test that metadata is converted to Bedrock tags."""
    pytest.importorskip("boto3")

    provider, mock_control, _ = _create_provider_with_mock_clients()

    job_arn = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/meta123"
    mock_control.create_model_invocation_job.return_value = {"jobArn": job_arn}
    mock_control.get_model_invocation_job.return_value = _make_mock_job(job_arn=job_arn)

    await provider._acreate_batch(
        input_file_path="s3://bucket/input.jsonl",
        endpoint="/v1/chat/completions",
        metadata={"env": "staging", "team": "ml"},
        role_arn="arn:aws:iam::123456789012:role/BatchRole",
        output_s3_uri="s3://bucket/output/",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
    )

    call_kwargs = mock_control.create_model_invocation_job.call_args[1]
    tags = call_kwargs["tags"]
    tag_dict = {t["key"]: t["value"] for t in tags}
    assert tag_dict == {"env": "staging", "team": "ml"}


@pytest.mark.asyncio
async def test_acreate_batch_auto_generates_job_name() -> None:
    """Test that a job name is auto-generated when not provided."""
    pytest.importorskip("boto3")

    provider, mock_control, _ = _create_provider_with_mock_clients()

    job_arn = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/auto123"
    mock_control.create_model_invocation_job.return_value = {"jobArn": job_arn}
    mock_control.get_model_invocation_job.return_value = _make_mock_job(job_arn=job_arn)

    await provider._acreate_batch(
        input_file_path="s3://bucket/input.jsonl",
        endpoint="/v1/chat/completions",
        role_arn="arn:aws:iam::123456789012:role/BatchRole",
        output_s3_uri="s3://bucket/output/",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
    )

    call_kwargs = mock_control.create_model_invocation_job.call_args[1]
    assert call_kwargs["jobName"].startswith("any-llm-batch-")


@pytest.mark.asyncio
async def test_aretrieve_batch() -> None:
    """Test retrieving a batch job."""
    pytest.importorskip("boto3")

    provider, mock_control, _ = _create_provider_with_mock_clients()

    job_arn = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/get123"
    mock_control.get_model_invocation_job.return_value = _make_mock_job(job_arn=job_arn, status="InProgress")

    result = await provider._aretrieve_batch(job_arn)

    assert result.id == job_arn
    assert result.status == "in_progress"
    mock_control.get_model_invocation_job.assert_called_once()
    call_kwargs = mock_control.get_model_invocation_job.call_args[1]
    assert call_kwargs["jobIdentifier"] == job_arn


@pytest.mark.asyncio
async def test_acancel_batch() -> None:
    """Test cancelling a batch job."""
    pytest.importorskip("boto3")

    provider, mock_control, _ = _create_provider_with_mock_clients()

    job_arn = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/cancel123"
    mock_control.stop_model_invocation_job.return_value = {}
    mock_control.get_model_invocation_job.return_value = _make_mock_job(job_arn=job_arn, status="Stopping")

    result = await provider._acancel_batch(job_arn)

    assert result.id == job_arn
    assert result.status == "cancelling"
    mock_control.stop_model_invocation_job.assert_called_once()
    stop_kwargs = mock_control.stop_model_invocation_job.call_args[1]
    assert stop_kwargs["jobIdentifier"] == job_arn


@pytest.mark.asyncio
async def test_alist_batches() -> None:
    """Test listing batch jobs."""
    pytest.importorskip("boto3")

    provider, mock_control, _ = _create_provider_with_mock_clients()

    mock_control.list_model_invocation_jobs.return_value = {
        "invocationJobSummaries": [
            _make_mock_job(job_arn="arn:job1", status="Completed"),
            _make_mock_job(job_arn="arn:job2", status="InProgress"),
        ]
    }

    result = await provider._alist_batches()

    assert len(result) == 2
    assert result[0].id == "arn:job1"
    assert result[0].status == "completed"
    assert result[1].id == "arn:job2"
    assert result[1].status == "in_progress"


@pytest.mark.asyncio
async def test_alist_batches_with_pagination() -> None:
    """Test listing batch jobs with after and limit parameters."""
    pytest.importorskip("boto3")

    provider, mock_control, _ = _create_provider_with_mock_clients()

    mock_control.list_model_invocation_jobs.return_value = {"invocationJobSummaries": []}

    await provider._alist_batches(after="next-token-123", limit=5)

    call_kwargs = mock_control.list_model_invocation_jobs.call_args[1]
    assert call_kwargs["nextToken"] == "next-token-123"
    assert call_kwargs["maxResults"] == 5


@pytest.mark.asyncio
async def test_aretrieve_batch_results_completed() -> None:
    """Test retrieving results from a completed batch job."""
    pytest.importorskip("boto3")

    provider, mock_control, mock_s3 = _create_provider_with_mock_clients()

    job_arn = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/results123"
    mock_control.get_model_invocation_job.return_value = _make_mock_job(
        job_arn=job_arn,
        status="Completed",
        input_s3_uri="s3://input-bucket/batch-input.jsonl",
        output_s3_uri="s3://output-bucket/results/",
    )

    output_record = json.dumps(
        {
            "recordId": "req-1",
            "modelOutput": {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Hello from Bedrock!"}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 10, "outputTokens": 5},
            },
        }
    )

    mock_body = MagicMock()
    mock_body.read.return_value = output_record.encode("utf-8")
    mock_s3.get_object.return_value = {"Body": mock_body}

    result = await provider._aretrieve_batch_results(job_arn)

    assert len(result.results) == 1
    assert result.results[0].custom_id == "req-1"
    assert result.results[0].result is not None
    assert result.results[0].error is None

    s3_call_kwargs = mock_s3.get_object.call_args[1]
    assert s3_call_kwargs["Bucket"] == "output-bucket"
    assert s3_call_kwargs["Key"] == "results/results123/batch-input.jsonl.out"


@pytest.mark.asyncio
async def test_aretrieve_batch_results_not_completed() -> None:
    """Test that BatchNotCompleteError is raised for incomplete batch."""
    pytest.importorskip("boto3")

    provider, mock_control, _ = _create_provider_with_mock_clients()

    job_arn = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/pending123"
    mock_control.get_model_invocation_job.return_value = _make_mock_job(job_arn=job_arn, status="InProgress")

    with pytest.raises(BatchNotCompleteError) as exc_info:
        await provider._aretrieve_batch_results(job_arn)

    assert exc_info.value.batch_id == job_arn
    assert exc_info.value.batch_status == "in_progress"


@pytest.mark.asyncio
async def test_aretrieve_batch_results_partially_completed() -> None:
    """Test retrieving results from a partially completed batch job."""
    pytest.importorskip("boto3")

    provider, mock_control, mock_s3 = _create_provider_with_mock_clients()

    job_arn = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/partial123"
    mock_control.get_model_invocation_job.return_value = _make_mock_job(
        job_arn=job_arn,
        status="PartiallyCompleted",
        input_s3_uri="s3://input-bucket/batch.jsonl",
        output_s3_uri="s3://output-bucket/out/",
        success_records=3,
        error_records=2,
    )

    output_lines = "\n".join(
        [
            json.dumps(
                {
                    "recordId": "req-1",
                    "modelOutput": {
                        "output": {"message": {"role": "assistant", "content": [{"text": "OK"}]}},
                        "stopReason": "end_turn",
                        "usage": {"inputTokens": 5, "outputTokens": 2},
                    },
                }
            ),
            json.dumps(
                {
                    "recordId": "req-2",
                    "error": {"errorCode": "ModelError", "errorMessage": "Model failed"},
                }
            ),
        ]
    )

    mock_body = MagicMock()
    mock_body.read.return_value = output_lines.encode("utf-8")
    mock_s3.get_object.return_value = {"Body": mock_body}

    result = await provider._aretrieve_batch_results(job_arn)

    assert len(result.results) == 2
    assert result.results[0].result is not None
    assert result.results[1].error is not None
    assert result.results[1].error.code == "ModelError"


@pytest.mark.asyncio
async def test_acreate_batch_invalid_input_s3_uri() -> None:
    """Test that an invalid input_file_path S3 URI raises InvalidRequestError early."""
    pytest.importorskip("boto3")

    provider, _, _ = _create_provider_with_mock_clients()

    with pytest.raises(InvalidRequestError, match="s3://"):
        await provider._acreate_batch(
            input_file_path="not-an-s3-uri/input.jsonl",
            endpoint="/v1/chat/completions",
            role_arn="arn:aws:iam::123456789012:role/BatchRole",
            output_s3_uri="s3://bucket/output/",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
        )


@pytest.mark.asyncio
async def test_acreate_batch_invalid_output_s3_uri() -> None:
    """Test that an invalid output_s3_uri raises InvalidRequestError early."""
    pytest.importorskip("boto3")

    provider, _, _ = _create_provider_with_mock_clients()

    with pytest.raises(InvalidRequestError, match="s3://"):
        await provider._acreate_batch(
            input_file_path="s3://bucket/input.jsonl",
            endpoint="/v1/chat/completions",
            role_arn="arn:aws:iam::123456789012:role/BatchRole",
            output_s3_uri="https://bucket/output/",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
        )


@pytest.mark.asyncio
async def test_aretrieve_batch_results_output_uri_without_trailing_slash() -> None:
    """Test that output S3 URIs without trailing slashes produce correct keys."""
    pytest.importorskip("boto3")

    provider, mock_control, mock_s3 = _create_provider_with_mock_clients()

    job_arn = "arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/slash123"
    mock_control.get_model_invocation_job.return_value = _make_mock_job(
        job_arn=job_arn,
        status="Completed",
        input_s3_uri="s3://input-bucket/batch-input.jsonl",
        output_s3_uri="s3://output-bucket/results",
    )

    output_record = json.dumps(
        {
            "recordId": "req-1",
            "modelOutput": {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Hello!"}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 10, "outputTokens": 5},
            },
        }
    )

    mock_body = MagicMock()
    mock_body.read.return_value = output_record.encode("utf-8")
    mock_s3.get_object.return_value = {"Body": mock_body}

    result = await provider._aretrieve_batch_results(job_arn)

    assert len(result.results) == 1
    s3_call_kwargs = mock_s3.get_object.call_args[1]
    assert s3_call_kwargs["Bucket"] == "output-bucket"
    assert s3_call_kwargs["Key"] == "results/slash123/batch-input.jsonl.out"


@pytest.mark.asyncio
async def test_control_client_does_not_use_runtime_endpoint() -> None:
    """Test that the control-plane client does not receive the runtime endpoint_url."""
    pytest.importorskip("boto3")
    from any_llm.providers.bedrock.bedrock import BedrockProvider

    with patch("any_llm.providers.bedrock.bedrock.boto3") as mock_boto3:
        mock_runtime = MagicMock()
        mock_boto3.client.return_value = mock_runtime
        mock_boto3.Session.return_value.get_credentials.return_value = MagicMock()

        provider = BedrockProvider(api_base="https://bedrock-runtime.us-east-1.amazonaws.com")

        mock_boto3.client.reset_mock()
        provider._get_bedrock_control_client()

        mock_boto3.client.assert_called_once_with("bedrock")
