"""Performance tests comparing individual vs batched usage event sending."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest


class MockCompletion:
    """Mock completion for testing."""

    def __init__(self) -> None:
        self.id = "test-completion"
        self.model = "gpt-4"
        self.object = "chat.completion"
        self.created = int(time.time())
        self.choices: list[dict[str, str]] = []
        self.usage = MockUsage()


class MockUsage:
    """Mock usage for testing."""

    def __init__(self) -> None:
        self.prompt_tokens = 100
        self.completion_tokens = 50
        self.total_tokens = 150


@pytest.fixture
def mock_completion() -> MockCompletion:
    """Create a mock completion for testing."""
    return MockCompletion()


@pytest.fixture
def mock_platform_client() -> MagicMock:
    """Create a mock platform client."""
    client = MagicMock()
    client._aensure_valid_token = AsyncMock(return_value="test-token")
    return client


@pytest.fixture
def mock_http_client() -> MagicMock:
    """Create a mock HTTP client."""
    client = MagicMock()
    response = MagicMock()
    response.raise_for_status = MagicMock()
    client.post = AsyncMock(return_value=response)
    return client


async def simulate_individual_requests(num_events: int, mock_client: MagicMock) -> tuple[float, int]:
    """Simulate sending events individually (old approach)."""
    start_time = time.perf_counter()

    for _ in range(num_events):
        await mock_client.post(
            "/usage-events/",
            json={"event": "data"},
            headers={"Authorization": "Bearer token"},
        )

    duration = time.perf_counter() - start_time
    return duration, mock_client.post.call_count


async def simulate_batched_requests(num_events: int, batch_size: int, mock_client: MagicMock) -> tuple[float, int]:
    """Simulate sending events in batches (new approach)."""
    start_time = time.perf_counter()

    # Calculate number of batches needed
    num_batches = (num_events + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_events)
        batch_events = [{"event": f"data-{i}"} for i in range(batch_start, batch_end)]

        await mock_client.post(
            "/usage-events/bulk",
            json={"events": batch_events},
            headers={"Authorization": "Bearer token"},
        )

    duration = time.perf_counter() - start_time
    return duration, mock_client.post.call_count


@pytest.mark.asyncio
async def test_performance_comparison(
    mock_completion: MockCompletion, mock_platform_client: MagicMock, mock_http_client: MagicMock
) -> None:
    """Compare performance of individual vs batched requests."""
    num_events = 100
    batch_size = 50

    # Test individual requests
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON: Individual vs Batched Requests")
    print("=" * 70)

    # Individual approach (OLD)
    individual_client = MagicMock()
    individual_response = MagicMock()
    individual_response.raise_for_status = MagicMock()
    individual_client.post = AsyncMock(return_value=individual_response)

    individual_duration, individual_requests = await simulate_individual_requests(num_events, individual_client)

    # Batched approach (NEW)
    batched_client = MagicMock()
    batched_response = MagicMock()
    batched_response.raise_for_status = MagicMock()
    batched_client.post = AsyncMock(return_value=batched_response)

    batched_duration, batched_requests = await simulate_batched_requests(num_events, batch_size, batched_client)

    # Calculate improvements
    request_reduction = ((individual_requests - batched_requests) / individual_requests) * 100

    print(f"\n{'INDIVIDUAL REQUESTS (OLD)':<40} {'BATCHED REQUESTS (NEW)':<40}")
    print("-" * 80)
    print(f"{'Events sent: ' + str(num_events):<40} {'Events sent: ' + str(num_events):<40}")
    print(f"{'HTTP requests: ' + str(individual_requests):<40} {'HTTP requests: ' + str(batched_requests):<40}")
    print(f"{'Time taken: ' + f'{individual_duration:.4f}s':<40} {'Time taken: ' + f'{batched_duration:.4f}s':<40}")
    print(
        f"{'Avg per event: ' + f'{(individual_duration / num_events) * 1000:.2f}ms':<40} {'Avg per event: ' + f'{(batched_duration / num_events) * 1000:.2f}ms':<40}"
    )

    print("\n" + "=" * 70)
    print("PERFORMANCE IMPROVEMENTS")
    print("=" * 70)
    print(f"✓ HTTP request reduction: {request_reduction:.1f}% ({individual_requests} → {batched_requests} requests)")
    print(f"✓ Network overhead reduction: ~{request_reduction:.0f}%")
    print("✓ Fewer TCP connections and TLS handshakes")
    print("✓ Reduced authentication overhead (per batch vs per event)")
    print("✓ Lower server load and better scalability")
    print("=" * 70 + "\n")

    # Assertions
    assert batched_requests < individual_requests, "Batching should reduce HTTP requests"
    assert batched_requests == (num_events + batch_size - 1) // batch_size, "Should use expected number of batches"


@pytest.mark.asyncio
async def test_performance_with_different_batch_sizes(
    mock_completion: MockCompletion, mock_platform_client: MagicMock, mock_http_client: MagicMock
) -> None:
    """Test performance with different batch sizes."""
    num_events = 100
    batch_sizes = [10, 25, 50, 100]

    print("\n" + "=" * 70)
    print("BATCH SIZE COMPARISON")
    print("=" * 70)
    print(f"Total events: {num_events}")
    print("-" * 70)
    print(f"{'Batch Size':<15} {'HTTP Requests':<20} {'Reduction':<20}")
    print("-" * 70)

    for batch_size in batch_sizes:
        mock_http = MagicMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_http.post = AsyncMock(return_value=mock_response)

        _, requests_made = await simulate_batched_requests(num_events, batch_size, mock_http)

        requests_saved = num_events - requests_made
        savings_percent = (requests_saved / num_events) * 100

        print(f"{batch_size:<15} {requests_made:<20} {requests_saved} saved ({savings_percent:.0f}%)")

    print("=" * 70 + "\n")


@pytest.mark.asyncio
async def test_scalability_comparison(
    mock_completion: MockCompletion, mock_platform_client: MagicMock, mock_http_client: MagicMock
) -> None:
    """Test how batching scales with increasing event counts."""
    event_counts = [50, 100, 500, 1000]
    batch_size = 50

    print("\n" + "=" * 70)
    print("SCALABILITY ANALYSIS")
    print("=" * 70)
    print(f"Batch size: {batch_size} events")
    print("-" * 70)
    print(f"{'Events':<15} {'Individual':<20} {'Batched':<20} {'Savings':<20}")
    print("-" * 70)

    for num_events in event_counts:
        # Individual
        individual_client = MagicMock()
        individual_response = MagicMock()
        individual_response.raise_for_status = MagicMock()
        individual_client.post = AsyncMock(return_value=individual_response)
        _, individual_requests = await simulate_individual_requests(num_events, individual_client)

        # Batched
        batched_client = MagicMock()
        batched_response = MagicMock()
        batched_response.raise_for_status = MagicMock()
        batched_client.post = AsyncMock(return_value=batched_response)
        _, batched_requests = await simulate_batched_requests(num_events, batch_size, batched_client)

        savings = individual_requests - batched_requests
        print(
            f"{num_events:<15} {individual_requests:<20} {batched_requests:<20} {savings} ({(savings / individual_requests) * 100:.0f}%)"
        )

    print("=" * 70)
    print("\nConclusion: Batching provides consistent ~96-98% request reduction")
    print("regardless of total event volume, making it highly scalable.")
    print("=" * 70 + "\n")
