import json
import tempfile
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from any_llm.exceptions import BatchNotCompleteError
from any_llm.types.batch import BatchResult


def _make_mock_message_batch(
    *,
    batch_id: str = "msgbatch_123",
    processing_status: str = "ended",
    processing: int = 0,
    succeeded: int = 5,
    errored: int = 0,
    canceled: int = 0,
    expired: int = 0,
) -> Mock:
    """Create a mock Anthropic MessageBatch object."""
    mock = Mock()
    mock.id = batch_id
    mock.processing_status = processing_status
    mock.request_counts = Mock()
    mock.request_counts.processing = processing
    mock.request_counts.succeeded = succeeded
    mock.request_counts.errored = errored
    mock.request_counts.canceled = canceled
    mock.request_counts.expired = expired
    mock.created_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    return mock


def test_convert_anthropic_batch_ended_all_succeeded() -> None:
    """Test conversion of ended batch with all successes."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.base import _convert_anthropic_batch_to_openai

    batch = _make_mock_message_batch(succeeded=5, errored=0)
    result = _convert_anthropic_batch_to_openai(batch)

    assert result.id == "msgbatch_123"
    assert result.status == "completed"
    assert result.request_counts is not None
    assert result.request_counts.total == 5
    assert result.request_counts.completed == 5
    assert result.request_counts.failed == 0


def test_convert_anthropic_batch_ended_with_failures() -> None:
    """Test conversion of ended batch with mixed results."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.base import _convert_anthropic_batch_to_openai

    batch = _make_mock_message_batch(succeeded=3, errored=1, canceled=1)
    result = _convert_anthropic_batch_to_openai(batch)

    assert result.status == "completed"
    assert result.request_counts is not None
    assert result.request_counts.total == 5
    assert result.request_counts.completed == 3
    assert result.request_counts.failed == 2


def test_convert_anthropic_batch_in_progress() -> None:
    """Test conversion of in_progress batch."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.base import _convert_anthropic_batch_to_openai

    batch = _make_mock_message_batch(processing_status="in_progress", processing=3, succeeded=2)
    result = _convert_anthropic_batch_to_openai(batch)

    assert result.status == "in_progress"
    assert result.request_counts is not None
    assert result.request_counts.total == 5


def test_convert_anthropic_batch_canceling() -> None:
    """Test conversion of canceling batch."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.base import _convert_anthropic_batch_to_openai

    batch = _make_mock_message_batch(processing_status="canceling", processing=2, succeeded=3)
    result = _convert_anthropic_batch_to_openai(batch)

    assert result.status == "cancelling"


def test_convert_anthropic_batch_canceled() -> None:
    """Test conversion of canceled batch."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.base import _convert_anthropic_batch_to_openai

    batch = _make_mock_message_batch(processing_status="canceled", succeeded=2, canceled=3)
    result = _convert_anthropic_batch_to_openai(batch)

    assert result.status == "cancelled"


def test_convert_anthropic_batch_expired() -> None:
    """Test conversion of expired batch."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.base import _convert_anthropic_batch_to_openai

    batch = _make_mock_message_batch(processing_status="expired", succeeded=1, expired=4)
    result = _convert_anthropic_batch_to_openai(batch)

    assert result.status == "expired"


def test_convert_anthropic_batch_unknown_status_logs_warning() -> None:
    """Test that unknown status logs a warning and defaults to in_progress."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.base import _convert_anthropic_batch_to_openai

    batch = _make_mock_message_batch(processing_status="some_new_status")
    with patch("any_llm.providers.anthropic.base.logger") as mock_logger:
        result = _convert_anthropic_batch_to_openai(batch)

    assert result.status == "in_progress"
    mock_logger.warning.assert_called_once()
    assert "Unknown Anthropic batch status" in mock_logger.warning.call_args[0][0]


def test_convert_anthropic_batch_request_counts_mapping() -> None:
    """Test that Anthropic multi-field counts map to OpenAI 3-field counts."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.base import _convert_anthropic_batch_to_openai

    batch = _make_mock_message_batch(
        processing=2,
        succeeded=5,
        errored=1,
        canceled=1,
        expired=1,
    )
    result = _convert_anthropic_batch_to_openai(batch)

    assert result.request_counts is not None
    assert result.request_counts.total == 10
    assert result.request_counts.completed == 5
    assert result.request_counts.failed == 3


@pytest.mark.asyncio
async def test_acreate_batch() -> None:
    """Test creating a batch with Anthropic provider."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.anthropic import AnthropicProvider

    with patch("any_llm.providers.anthropic.anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="test-key")

        mock_result = _make_mock_message_batch()
        provider.client.messages.batches.create = AsyncMock(return_value=mock_result)  # type: ignore[method-assign]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "custom_id": "req-1",
                        "body": {
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 100,
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

            assert result.id == "msgbatch_123"
            assert result.status == "completed"
            provider.client.messages.batches.create.assert_called_once()

            call_kwargs = provider.client.messages.batches.create.call_args[1]
            assert len(call_kwargs["requests"]) == 1
            assert call_kwargs["requests"][0]["custom_id"] == "req-1"
            assert call_kwargs["requests"][0]["params"]["model"] == "claude-sonnet-4-20250514"
        finally:
            import os

            os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_aretrieve_batch_results_completed_mixed() -> None:
    """Test retrieving batch results with mixed successes and failures."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.anthropic import AnthropicProvider

    with patch("any_llm.providers.anthropic.anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="test-key")

        mock_batch = _make_mock_message_batch(processing_status="ended", succeeded=1, errored=1)
        provider.client.messages.batches.retrieve = AsyncMock(return_value=mock_batch)  # type: ignore[method-assign]

        success_entry = Mock()
        success_entry.custom_id = "req-1"
        success_entry.result = Mock()
        success_entry.result.type = "succeeded"

        mock_message = Mock()
        mock_message.id = "msg_123"
        mock_message.model = "claude-sonnet-4-20250514"
        mock_message.stop_reason = "end_turn"
        mock_message.content = [Mock(type="text", text="Hello!")]
        mock_message.usage = Mock(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        mock_message.created_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        success_entry.result.message = mock_message

        error_entry = Mock()
        error_entry.custom_id = "req-2"
        error_entry.result = Mock()
        error_entry.result.type = "errored"
        error_entry.result.error = Mock()
        error_entry.result.error.error = Mock()
        error_entry.result.error.error.type = "invalid_request"
        error_entry.result.error.error.message = "Bad request"

        class MockAsyncIter:
            def __init__(self, entries: list[Mock]) -> None:
                self._entries = entries

            def __aiter__(self) -> "MockAsyncIter":
                self._index = 0
                return self

            async def __anext__(self) -> Mock:
                if self._index >= len(self._entries):
                    raise StopAsyncIteration
                entry = self._entries[self._index]
                self._index += 1
                return entry

        provider.client.messages.batches.results = AsyncMock(return_value=MockAsyncIter([success_entry, error_entry]))  # type: ignore[method-assign]

        result = await provider._aretrieve_batch_results("msgbatch_123")

        assert isinstance(result, BatchResult)
        assert len(result.results) == 2
        assert result.results[0].custom_id == "req-1"
        assert result.results[0].result is not None
        assert result.results[0].error is None
        assert result.results[1].custom_id == "req-2"
        assert result.results[1].result is None
        assert result.results[1].error is not None
        assert result.results[1].error.code == "invalid_request"


@pytest.mark.asyncio
async def test_aretrieve_batch_results_not_completed() -> None:
    """Test that BatchNotCompleteError is raised for non-ended batch."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.anthropic import AnthropicProvider

    with patch("any_llm.providers.anthropic.anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="test-key")

        mock_batch = _make_mock_message_batch(processing_status="in_progress", processing=5, succeeded=0)
        provider.client.messages.batches.retrieve = AsyncMock(return_value=mock_batch)  # type: ignore[method-assign]

        with pytest.raises(BatchNotCompleteError) as exc_info:
            await provider._aretrieve_batch_results("msgbatch_not_done")

        assert exc_info.value.batch_id == "msgbatch_not_done"
        assert exc_info.value.batch_status == "in_progress"


def test_convert_anthropic_batch_created_at_none() -> None:
    """Test conversion when created_at is None falls back to 0."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.base import _convert_anthropic_batch_to_openai

    batch = _make_mock_message_batch()
    batch.created_at = None
    result = _convert_anthropic_batch_to_openai(batch)

    assert result.created_at == 0


@pytest.mark.asyncio
async def test_aretrieve_batch_delegates() -> None:
    """Test _aretrieve_batch calls Anthropic API and converts result."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.anthropic import AnthropicProvider

    with patch("any_llm.providers.anthropic.anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="test-key")

        mock_batch = _make_mock_message_batch(processing_status="in_progress", processing=3, succeeded=2)
        provider.client.messages.batches.retrieve = AsyncMock(return_value=mock_batch)  # type: ignore[method-assign]

        result = await provider._aretrieve_batch("msgbatch_123")

        assert result.id == "msgbatch_123"
        assert result.status == "in_progress"
        provider.client.messages.batches.retrieve.assert_called_once_with("msgbatch_123")


@pytest.mark.asyncio
async def test_acancel_batch_delegates() -> None:
    """Test _acancel_batch calls Anthropic API and converts result."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.anthropic import AnthropicProvider

    with patch("any_llm.providers.anthropic.anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="test-key")

        mock_batch = _make_mock_message_batch(processing_status="canceling", processing=2, succeeded=3)
        provider.client.messages.batches.cancel = AsyncMock(return_value=mock_batch)  # type: ignore[method-assign]

        result = await provider._acancel_batch("msgbatch_123")

        assert result.status == "cancelling"
        provider.client.messages.batches.cancel.assert_called_once_with("msgbatch_123")


@pytest.mark.asyncio
async def test_alist_batches_delegates() -> None:
    """Test _alist_batches calls Anthropic API with correct params."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.anthropic import AnthropicProvider

    with patch("any_llm.providers.anthropic.anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="test-key")

        mock_batch = _make_mock_message_batch()
        mock_result = Mock()
        mock_result.data = [mock_batch]
        provider.client.messages.batches.list = AsyncMock(return_value=mock_result)  # type: ignore[method-assign]

        result = await provider._alist_batches(after="msgbatch_100", limit=5)

        assert len(result) == 1
        assert result[0].id == "msgbatch_123"
        provider.client.messages.batches.list.assert_called_once_with(after_id="msgbatch_100", limit=5)


@pytest.mark.asyncio
async def test_alist_batches_no_params() -> None:
    """Test _alist_batches without after or limit passes no extra kwargs."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.anthropic import AnthropicProvider

    with patch("any_llm.providers.anthropic.anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="test-key")

        mock_result = Mock()
        mock_result.data = []
        provider.client.messages.batches.list = AsyncMock(return_value=mock_result)  # type: ignore[method-assign]

        result = await provider._alist_batches()

        assert result == []
        provider.client.messages.batches.list.assert_called_once_with()


@pytest.mark.asyncio
async def test_acreate_batch_skips_empty_lines() -> None:
    """Test _acreate_batch skips empty lines in the JSONL input file."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.anthropic import AnthropicProvider

    with patch("any_llm.providers.anthropic.anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="test-key")

        mock_result = _make_mock_message_batch()
        provider.client.messages.batches.create = AsyncMock(return_value=mock_result)  # type: ignore[method-assign]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "custom_id": "req-1",
                        "body": {
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 100,
                            "messages": [{"role": "user", "content": "Hello"}],
                        },
                    }
                )
                + "\n"
            )
            f.write("  \n")
            f.write(
                json.dumps(
                    {
                        "custom_id": "req-2",
                        "body": {
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 100,
                            "messages": [{"role": "user", "content": "World"}],
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

            assert result.id == "msgbatch_123"
            call_kwargs = provider.client.messages.batches.create.call_args[1]
            assert len(call_kwargs["requests"]) == 2
        finally:
            import os

            os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_acreate_batch_with_optional_params() -> None:
    """Test _acreate_batch passes temperature, top_p, and system when present."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.anthropic import AnthropicProvider

    with patch("any_llm.providers.anthropic.anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="test-key")

        mock_result = _make_mock_message_batch()
        provider.client.messages.batches.create = AsyncMock(return_value=mock_result)  # type: ignore[method-assign]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(
                json.dumps(
                    {
                        "custom_id": "req-1",
                        "body": {
                            "model": "claude-sonnet-4-20250514",
                            "max_tokens": 100,
                            "messages": [{"role": "user", "content": "Hello"}],
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "system": "You are helpful.",
                        },
                    }
                )
                + "\n"
            )
            tmp_path = f.name

        try:
            await provider._acreate_batch(
                input_file_path=tmp_path,
                endpoint="/v1/chat/completions",
            )

            call_kwargs = provider.client.messages.batches.create.call_args[1]
            params = call_kwargs["requests"][0]["params"]
            assert params["temperature"] == 0.7
            assert params["top_p"] == 0.9
            assert params["system"] == "You are helpful."
        finally:
            import os

            os.unlink(tmp_path)


@pytest.mark.asyncio
async def test_aretrieve_batch_results_other_result_type() -> None:
    """Test that result types other than 'succeeded' or 'errored' produce an error item."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.anthropic import AnthropicProvider

    with patch("any_llm.providers.anthropic.anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="test-key")

        mock_batch = _make_mock_message_batch(processing_status="ended", succeeded=0, canceled=1)
        provider.client.messages.batches.retrieve = AsyncMock(return_value=mock_batch)  # type: ignore[method-assign]

        canceled_entry = Mock()
        canceled_entry.custom_id = "req-canceled"
        canceled_entry.result = Mock()
        canceled_entry.result.type = "canceled"

        class MockAsyncIter:
            def __init__(self, entries: list[Mock]) -> None:
                self._entries = entries

            def __aiter__(self) -> "MockAsyncIter":
                self._index = 0
                return self

            async def __anext__(self) -> Mock:
                if self._index >= len(self._entries):
                    raise StopAsyncIteration
                entry = self._entries[self._index]
                self._index += 1
                return entry

        provider.client.messages.batches.results = AsyncMock(return_value=MockAsyncIter([canceled_entry]))  # type: ignore[method-assign]

        result = await provider._aretrieve_batch_results("msgbatch_canceled")

        assert len(result.results) == 1
        assert result.results[0].custom_id == "req-canceled"
        assert result.results[0].error is not None
        assert result.results[0].error.code == "canceled"
        assert "Request canceled" in result.results[0].error.message


@pytest.mark.asyncio
async def test_aretrieve_batch_results_errored_with_null_error() -> None:
    """Test errored entry where error or error.error is None falls back to 'unknown'."""
    pytest.importorskip("anthropic")
    from any_llm.providers.anthropic.anthropic import AnthropicProvider

    with patch("any_llm.providers.anthropic.anthropic.AsyncAnthropic"):
        provider = AnthropicProvider(api_key="test-key")

        mock_batch = _make_mock_message_batch(processing_status="ended", succeeded=0, errored=1)
        provider.client.messages.batches.retrieve = AsyncMock(return_value=mock_batch)  # type: ignore[method-assign]

        error_entry = Mock()
        error_entry.custom_id = "req-err"
        error_entry.result = Mock()
        error_entry.result.type = "errored"
        error_entry.result.error = None

        class MockAsyncIter:
            def __init__(self, entries: list[Mock]) -> None:
                self._entries = entries

            def __aiter__(self) -> "MockAsyncIter":
                self._index = 0
                return self

            async def __anext__(self) -> Mock:
                if self._index >= len(self._entries):
                    raise StopAsyncIteration
                entry = self._entries[self._index]
                self._index += 1
                return entry

        provider.client.messages.batches.results = AsyncMock(return_value=MockAsyncIter([error_entry]))  # type: ignore[method-assign]

        result = await provider._aretrieve_batch_results("msgbatch_err")

        assert len(result.results) == 1
        assert result.results[0].error is not None
        assert result.results[0].error.code == "unknown"
        assert result.results[0].error.message == "Unknown error"
