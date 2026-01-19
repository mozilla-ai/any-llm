"""Tests for usage event batch queue functionality."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("any_llm_platform_client", reason="any_llm_platform_client not installed")


class MockCompletion:
    """Mock completion for testing."""

    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.id = "test-completion"
        self.model = "gpt-4"
        self.usage = MockUsage(prompt_tokens, completion_tokens)


class MockUsage:
    """Mock usage for testing."""

    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


@pytest.fixture
def mock_platform_client() -> MagicMock:
    """Create a mock platform client."""
    client = MagicMock()
    client._aensure_valid_token = AsyncMock(return_value="test-token")
    return client


@pytest.fixture
def mock_http_client() -> MagicMock:
    """Create a mock HTTP client that tracks calls."""
    client = MagicMock()
    response = MagicMock()
    response.raise_for_status = MagicMock()
    client.post = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_batch_size_triggers_flush(mock_platform_client: MagicMock, mock_http_client: MagicMock) -> None:
    """Test that reaching batch size triggers automatic flush."""
    from any_llm.providers.platform.batch_queue import UsageEventBatchQueue

    batch_size = 3
    queue = UsageEventBatchQueue(
        platform_client=mock_platform_client,
        http_client=mock_http_client,
        batch_size=batch_size,
        flush_interval=100.0,  # Very long to ensure size-based flush
    )

    # Enqueue events one by one
    for i in range(batch_size - 1):
        await queue.enqueue(
            any_llm_key="test-key",
            provider="openai",
            completion=MockCompletion(),  # type: ignore[arg-type]
            provider_key_id=f"key-{i}",
        )

    # Should not have flushed yet
    assert mock_http_client.post.call_count == 0

    # Add one more to reach batch size
    await queue.enqueue(
        any_llm_key="test-key",
        provider="openai",
        completion=MockCompletion(),  # type: ignore[arg-type]
        provider_key_id="key-final",
    )

    # Should have flushed automatically
    assert mock_http_client.post.call_count == 1

    # Verify the batch was sent
    call_args = mock_http_client.post.call_args
    assert "/usage-events/bulk" in call_args[0][0]
    payload = call_args[1]["json"]
    assert len(payload["events"]) == batch_size

    await queue.shutdown()


@pytest.mark.asyncio
async def test_time_window_triggers_flush(mock_platform_client: MagicMock, mock_http_client: MagicMock) -> None:
    """Test that time window expiration triggers flush."""
    from any_llm.providers.platform.batch_queue import UsageEventBatchQueue

    flush_interval = 0.2  # 200ms
    queue = UsageEventBatchQueue(
        platform_client=mock_platform_client,
        http_client=mock_http_client,
        batch_size=100,  # Very large to ensure time-based flush
        flush_interval=flush_interval,
    )

    # Enqueue a few events (less than batch size)
    num_events = 3
    for i in range(num_events):
        await queue.enqueue(
            any_llm_key="test-key",
            provider="openai",
            completion=MockCompletion(),  # type: ignore[arg-type]
            provider_key_id=f"key-{i}",
        )

    # Should not have flushed yet
    assert mock_http_client.post.call_count == 0

    # Wait for time window to expire
    await asyncio.sleep(flush_interval + 0.1)

    # Should have flushed by time
    assert mock_http_client.post.call_count == 1

    # Verify the batch was sent
    call_args = mock_http_client.post.call_args
    payload = call_args[1]["json"]
    assert len(payload["events"]) == num_events

    await queue.shutdown()


@pytest.mark.asyncio
async def test_queue_groups_by_api_key(mock_platform_client: MagicMock, mock_http_client: MagicMock) -> None:
    """Test that events are grouped by API key for separate batches."""
    from any_llm.providers.platform.batch_queue import UsageEventBatchQueue

    batch_size = 2
    queue = UsageEventBatchQueue(
        platform_client=mock_platform_client,
        http_client=mock_http_client,
        batch_size=batch_size,
        flush_interval=100.0,
    )

    # Enqueue events with different API keys
    await queue.enqueue(
        any_llm_key="key-1",
        provider="openai",
        completion=MockCompletion(),  # type: ignore[arg-type]
        provider_key_id="provider-key-1",
    )
    await queue.enqueue(
        any_llm_key="key-2",
        provider="openai",
        completion=MockCompletion(),  # type: ignore[arg-type]
        provider_key_id="provider-key-2",
    )

    # Should flush (batch size reached)
    assert mock_http_client.post.call_count == 2  # One per API key

    await queue.shutdown()


@pytest.mark.asyncio
async def test_manual_flush(mock_platform_client: MagicMock, mock_http_client: MagicMock) -> None:
    """Test manual flush works correctly."""
    from any_llm.providers.platform.batch_queue import UsageEventBatchQueue

    queue = UsageEventBatchQueue(
        platform_client=mock_platform_client,
        http_client=mock_http_client,
        batch_size=100,
        flush_interval=100.0,
    )

    # Enqueue some events
    num_events = 5
    for i in range(num_events):
        await queue.enqueue(
            any_llm_key="test-key",
            provider="openai",
            completion=MockCompletion(),  # type: ignore[arg-type]
            provider_key_id=f"key-{i}",
        )

    # Should not have auto-flushed
    assert mock_http_client.post.call_count == 0

    # Manual flush
    await queue.flush()

    # Should have flushed
    assert mock_http_client.post.call_count == 1
    call_args = mock_http_client.post.call_args
    payload = call_args[1]["json"]
    assert len(payload["events"]) == num_events

    await queue.shutdown()


@pytest.mark.asyncio
async def test_shutdown_flushes_remaining_events(mock_platform_client: MagicMock, mock_http_client: MagicMock) -> None:
    """Test that shutdown flushes remaining events."""
    from any_llm.providers.platform.batch_queue import UsageEventBatchQueue

    queue = UsageEventBatchQueue(
        platform_client=mock_platform_client,
        http_client=mock_http_client,
        batch_size=100,
        flush_interval=100.0,
    )

    # Enqueue some events
    num_events = 5
    for i in range(num_events):
        await queue.enqueue(
            any_llm_key="test-key",
            provider="openai",
            completion=MockCompletion(),  # type: ignore[arg-type]
            provider_key_id=f"key-{i}",
        )

    # Should not have auto-flushed
    assert mock_http_client.post.call_count == 0

    # Shutdown (should flush)
    await queue.shutdown()

    # Should have flushed during shutdown
    assert mock_http_client.post.call_count == 1
    call_args = mock_http_client.post.call_args
    payload = call_args[1]["json"]
    assert len(payload["events"]) == num_events


@pytest.mark.asyncio
async def test_empty_queue_does_not_send_request(mock_platform_client: MagicMock, mock_http_client: MagicMock) -> None:
    """Test that empty queue doesn't send unnecessary requests."""
    from any_llm.providers.platform.batch_queue import UsageEventBatchQueue

    queue = UsageEventBatchQueue(
        platform_client=mock_platform_client,
        http_client=mock_http_client,
        batch_size=10,
        flush_interval=100.0,
    )

    # Flush empty queue
    await queue.flush()

    # Should not send any request
    assert mock_http_client.post.call_count == 0

    await queue.shutdown()


@pytest.mark.asyncio
async def test_events_without_usage_are_skipped(mock_platform_client: MagicMock, mock_http_client: MagicMock) -> None:
    """Test that events without usage data are skipped."""
    from any_llm.providers.platform.batch_queue import UsageEventBatchQueue

    queue = UsageEventBatchQueue(
        platform_client=mock_platform_client,
        http_client=mock_http_client,
        batch_size=10,
        flush_interval=100.0,
    )

    # Create completion without usage
    completion_no_usage = MockCompletion()
    completion_no_usage.usage = None  # type: ignore[assignment]

    # Try to enqueue
    await queue.enqueue(
        any_llm_key="test-key",
        provider="openai",
        completion=completion_no_usage,  # type: ignore[arg-type]
        provider_key_id="key-1",
    )

    # Should not add to queue (returns early)
    await queue.flush()

    # Should not send any request
    assert mock_http_client.post.call_count == 0

    await queue.shutdown()


@pytest.mark.asyncio
async def test_multiple_batches_from_size(mock_platform_client: MagicMock, mock_http_client: MagicMock) -> None:
    """Test that multiple batches are sent when events exceed batch size."""
    from any_llm.providers.platform.batch_queue import UsageEventBatchQueue

    batch_size = 3
    queue = UsageEventBatchQueue(
        platform_client=mock_platform_client,
        http_client=mock_http_client,
        batch_size=batch_size,
        flush_interval=100.0,
    )

    # Enqueue 2 * batch_size events
    num_events = batch_size * 2
    for i in range(num_events):
        await queue.enqueue(
            any_llm_key="test-key",
            provider="openai",
            completion=MockCompletion(),  # type: ignore[arg-type]
            provider_key_id=f"key-{i}",
        )

    # Should have flushed twice (once per batch)
    assert mock_http_client.post.call_count == 2

    # Verify batch sizes
    for call in mock_http_client.post.call_args_list:
        payload = call[1]["json"]
        assert len(payload["events"]) == batch_size

    await queue.shutdown()


@pytest.mark.asyncio
async def test_background_flush_task_runs(mock_platform_client: MagicMock, mock_http_client: MagicMock) -> None:
    """Test that background flush task is created and runs."""
    from any_llm.providers.platform.batch_queue import UsageEventBatchQueue

    flush_interval = 0.15
    queue = UsageEventBatchQueue(
        platform_client=mock_platform_client,
        http_client=mock_http_client,
        batch_size=100,
        flush_interval=flush_interval,
    )

    # Enqueue one event to start background task
    await queue.enqueue(
        any_llm_key="test-key",
        provider="openai",
        completion=MockCompletion(),  # type: ignore[arg-type]
        provider_key_id="key-1",
    )

    # Background task should be running
    assert queue._flush_task is not None
    assert not queue._flush_task.done()

    # Wait for background flush
    await asyncio.sleep(flush_interval + 0.1)

    # Should have flushed
    assert mock_http_client.post.call_count >= 1

    await queue.shutdown()


@pytest.mark.asyncio
async def test_payload_includes_performance_metrics(
    mock_platform_client: MagicMock, mock_http_client: MagicMock
) -> None:
    """Test that performance metrics are included in payload."""
    from any_llm.providers.platform.batch_queue import UsageEventBatchQueue

    queue = UsageEventBatchQueue(
        platform_client=mock_platform_client,
        http_client=mock_http_client,
        batch_size=1,
        flush_interval=100.0,
    )

    # Enqueue with performance metrics
    await queue.enqueue(
        any_llm_key="test-key",
        provider="openai",
        completion=MockCompletion(),  # type: ignore[arg-type]
        provider_key_id="key-1",
        time_to_first_token_ms=150.5,
        time_to_last_token_ms=2500.0,
        total_duration_ms=2650.5,
        tokens_per_second=18.87,
        chunks_received=25,
        avg_chunk_size=2.0,
        inter_chunk_latency_variance_ms=10.5,
    )

    # Should have flushed
    assert mock_http_client.post.call_count == 1

    # Verify performance metrics in payload
    call_args = mock_http_client.post.call_args
    payload = call_args[1]["json"]
    event = payload["events"][0]

    assert "performance" in event["data"]
    perf = event["data"]["performance"]
    assert perf["time_to_first_token_ms"] == 150.5
    assert perf["time_to_last_token_ms"] == 2500.0
    assert perf["total_duration_ms"] == 2650.5
    assert perf["tokens_per_second"] == 18.87
    assert perf["chunks_received"] == 25
    assert perf["avg_chunk_size"] == 2.0
    assert perf["inter_chunk_latency_variance_ms"] == 10.5

    await queue.shutdown()


@pytest.mark.asyncio
async def test_payload_includes_client_name(mock_platform_client: MagicMock, mock_http_client: MagicMock) -> None:
    """Test that client name is included in payload when provided."""
    from any_llm.providers.platform.batch_queue import UsageEventBatchQueue

    queue = UsageEventBatchQueue(
        platform_client=mock_platform_client,
        http_client=mock_http_client,
        batch_size=1,
        flush_interval=100.0,
    )

    # Enqueue with client name
    await queue.enqueue(
        any_llm_key="test-key",
        provider="openai",
        completion=MockCompletion(),  # type: ignore[arg-type]
        provider_key_id="key-1",
        client_name="test-client",
    )

    # Should have flushed
    assert mock_http_client.post.call_count == 1

    # Verify client name in payload
    call_args = mock_http_client.post.call_args
    payload = call_args[1]["json"]
    event = payload["events"][0]

    assert event["client_name"] == "test-client"

    await queue.shutdown()


@pytest.mark.asyncio
async def test_concurrent_enqueue_operations(mock_platform_client: MagicMock, mock_http_client: MagicMock) -> None:
    """Test that concurrent enqueue operations are handled safely."""
    from any_llm.providers.platform.batch_queue import UsageEventBatchQueue

    queue = UsageEventBatchQueue(
        platform_client=mock_platform_client,
        http_client=mock_http_client,
        batch_size=100,
        flush_interval=100.0,
    )

    # Enqueue many events concurrently
    num_events = 50
    tasks = [
        queue.enqueue(
            any_llm_key="test-key",
            provider="openai",
            completion=MockCompletion(),  # type: ignore[arg-type]
            provider_key_id=f"key-{i}",
        )
        for i in range(num_events)
    ]

    await asyncio.gather(*tasks)

    # Flush and verify all events were queued
    await queue.flush()

    assert mock_http_client.post.call_count == 1
    call_args = mock_http_client.post.call_args
    payload = call_args[1]["json"]
    assert len(payload["events"]) == num_events

    await queue.shutdown()


@pytest.mark.asyncio
async def test_completion_without_usage_returns_empty_payload(
    mock_platform_client: MagicMock, mock_http_client: MagicMock
) -> None:
    """Test that completion without usage returns empty payload."""
    from any_llm.providers.platform.utils import build_usage_event_payload

    # Create completion without usage
    completion_no_usage = MagicMock()
    completion_no_usage.id = "test-id"
    completion_no_usage.model = "gpt-4"
    completion_no_usage.usage = None

    payload = build_usage_event_payload(
        provider="openai",
        completion=completion_no_usage,
        provider_key_id="key-1",
    )

    assert payload == {}


@pytest.mark.asyncio
async def test_batch_send_with_empty_payloads_skips_request(
    mock_platform_client: MagicMock, mock_http_client: MagicMock
) -> None:
    """Test that batch with only empty payloads skips HTTP request."""
    from any_llm.providers.platform.batch_queue import UsageEventBatchQueue

    queue = UsageEventBatchQueue(
        platform_client=mock_platform_client,
        http_client=mock_http_client,
        batch_size=10,
        flush_interval=100.0,
    )

    # Create completions without usage
    for i in range(3):
        completion_no_usage = MagicMock()
        completion_no_usage.id = f"test-{i}"
        completion_no_usage.model = "gpt-4"
        completion_no_usage.usage = None

        await queue.enqueue(
            any_llm_key="test-key",
            provider="openai",
            completion=completion_no_usage,
            provider_key_id=f"key-{i}",
        )

    await queue.flush()

    # Should not have made any HTTP requests since all payloads were empty
    assert mock_http_client.post.call_count == 0

    await queue.shutdown()


@pytest.mark.asyncio
async def test_send_batch_error_handling(mock_platform_client: MagicMock, mock_http_client: MagicMock) -> None:
    """Test error handling when sending batch fails."""
    from any_llm.providers.platform.batch_queue import UsageEventBatchQueue

    # Make HTTP client raise an error
    mock_http_client.post = AsyncMock(side_effect=Exception("Network error"))

    queue = UsageEventBatchQueue(
        platform_client=mock_platform_client,
        http_client=mock_http_client,
        batch_size=2,
        flush_interval=100.0,
    )

    await queue.enqueue(
        any_llm_key="test-key",
        provider="openai",
        completion=MockCompletion(),  # type: ignore[arg-type]
        provider_key_id="key-1",
    )

    await queue.enqueue(
        any_llm_key="test-key",
        provider="openai",
        completion=MockCompletion(),  # type: ignore[arg-type]
        provider_key_id="key-2",
    )

    # Should reach batch size and attempt to send (which will fail)
    # The error should be logged but not raised
    await asyncio.sleep(0.1)  # Give time for background flush

    await queue.shutdown()

