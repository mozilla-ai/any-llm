"""Batching queue for usage events to reduce HTTP request overhead."""

from __future__ import annotations

import asyncio
import atexit
import time
import uuid
from typing import TYPE_CHECKING, Any

import httpx  # noqa: TC002
from any_llm_platform_client import AnyLLMPlatformClient  # noqa: TC002

from any_llm.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from any_llm.types.completion import ChatCompletion


class UsageEventBatch:
    """Represents a batch of usage events to be sent."""

    def __init__(
        self,
        any_llm_key: str,
        provider: str,
        completion: ChatCompletion,
        provider_key_id: str,
        client_name: str | None = None,
        time_to_first_token_ms: float | None = None,
        time_to_last_token_ms: float | None = None,
        total_duration_ms: float | None = None,
        tokens_per_second: float | None = None,
        chunks_received: int | None = None,
        avg_chunk_size: float | None = None,
        inter_chunk_latency_variance_ms: float | None = None,
    ):
        self.id = str(uuid.uuid4())
        self.any_llm_key = any_llm_key
        self.provider = provider
        self.completion = completion
        self.provider_key_id = provider_key_id
        self.client_name = client_name
        self.time_to_first_token_ms = time_to_first_token_ms
        self.time_to_last_token_ms = time_to_last_token_ms
        self.total_duration_ms = total_duration_ms
        self.tokens_per_second = tokens_per_second
        self.chunks_received = chunks_received
        self.avg_chunk_size = avg_chunk_size
        self.inter_chunk_latency_variance_ms = inter_chunk_latency_variance_ms

    def to_payload(self) -> dict[str, Any]:
        """Convert to API payload format."""
        if self.completion.usage is None:
            return {}

        data: dict[str, Any] = {
            "input_tokens": str(self.completion.usage.prompt_tokens),
            "output_tokens": str(self.completion.usage.completion_tokens),
        }

        performance: dict[str, float | int] = {}
        if self.time_to_first_token_ms is not None:
            performance["time_to_first_token_ms"] = self.time_to_first_token_ms
        if self.time_to_last_token_ms is not None:
            performance["time_to_last_token_ms"] = self.time_to_last_token_ms
        if self.total_duration_ms is not None:
            performance["total_duration_ms"] = self.total_duration_ms
        if self.tokens_per_second is not None:
            performance["tokens_per_second"] = self.tokens_per_second
        if self.chunks_received is not None:
            performance["chunks_received"] = self.chunks_received
        if self.avg_chunk_size is not None:
            performance["avg_chunk_size"] = self.avg_chunk_size
        if self.inter_chunk_latency_variance_ms is not None:
            performance["inter_chunk_latency_variance_ms"] = self.inter_chunk_latency_variance_ms

        if performance:
            data["performance"] = performance

        payload = {
            "id": self.id,
            "provider_key_id": self.provider_key_id,
            "provider": self.provider,
            "model": self.completion.model,
            "data": data,
        }
        if self.client_name:
            payload["client_name"] = self.client_name

        return payload


class UsageEventBatchQueue:
    """Manages batching of usage events to reduce HTTP overhead.

    Events are batched by size (default: 50 events) or time window (default: 5 seconds),
    whichever comes first. This significantly reduces the number of HTTP requests while
    maintaining reasonable latency for analytics.
    """

    def __init__(
        self,
        platform_client: AnyLLMPlatformClient,
        http_client: httpx.AsyncClient,
        batch_size: int = 50,
        flush_interval: float = 5.0,
    ):
        """Initialize the batch queue.

        Args:
            platform_client: Platform client for authentication.
            http_client: HTTP client for making requests.
            batch_size: Maximum number of events per batch (default: 50).
            flush_interval: Maximum seconds to wait before flushing (default: 5.0).
        """
        self.platform_client = platform_client
        self.http_client = http_client
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._queue: list[UsageEventBatch] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task[None] | None = None
        self._last_flush_time = time.monotonic()
        self._shutdown = False

    async def enqueue(
        self,
        any_llm_key: str,
        provider: str,
        completion: ChatCompletion,
        provider_key_id: str,
        client_name: str | None = None,
        time_to_first_token_ms: float | None = None,
        time_to_last_token_ms: float | None = None,
        total_duration_ms: float | None = None,
        tokens_per_second: float | None = None,
        chunks_received: int | None = None,
        avg_chunk_size: float | None = None,
        inter_chunk_latency_variance_ms: float | None = None,
    ) -> None:
        """Add a usage event to the batch queue.

        Args:
            any_llm_key: The Any LLM platform key.
            provider: The name of the LLM provider.
            completion: The LLM response.
            provider_key_id: The unique identifier for the provider key.
            client_name: Optional name of the client.
            time_to_first_token_ms: Time to first token in milliseconds.
            time_to_last_token_ms: Time to last token in milliseconds.
            total_duration_ms: Total request duration in milliseconds.
            tokens_per_second: Average token generation throughput.
            chunks_received: Number of chunks received.
            avg_chunk_size: Average tokens per chunk.
            inter_chunk_latency_variance_ms: Inter-chunk latency variance.
        """
        if completion.usage is None:
            return

        event = UsageEventBatch(
            any_llm_key=any_llm_key,
            provider=provider,
            completion=completion,
            provider_key_id=provider_key_id,
            client_name=client_name,
            time_to_first_token_ms=time_to_first_token_ms,
            time_to_last_token_ms=time_to_last_token_ms,
            total_duration_ms=total_duration_ms,
            tokens_per_second=tokens_per_second,
            chunks_received=chunks_received,
            avg_chunk_size=avg_chunk_size,
            inter_chunk_latency_variance_ms=inter_chunk_latency_variance_ms,
        )

        async with self._lock:
            self._queue.append(event)

            # Flush if batch size reached
            if len(self._queue) >= self.batch_size:
                await self._flush_internal()

            # Start background flush task if not running
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._background_flush())

    async def _background_flush(self) -> None:
        """Background task that flushes the queue periodically."""
        while not self._shutdown:
            await asyncio.sleep(self.flush_interval)

            async with self._lock:
                if self._queue and (time.monotonic() - self._last_flush_time >= self.flush_interval):
                    await self._flush_internal()

    async def _flush_internal(self) -> None:
        """Internal flush method (must be called with lock held)."""
        if not self._queue:
            return

        # Group events by any_llm_key (different projects may have different keys)
        events_by_key: dict[str, list[UsageEventBatch]] = {}
        for event in self._queue:
            if event.any_llm_key not in events_by_key:
                events_by_key[event.any_llm_key] = []
            events_by_key[event.any_llm_key].append(event)

        # Clear queue
        self._queue.clear()
        self._last_flush_time = time.monotonic()

        # Send batches for each key
        for any_llm_key, events in events_by_key.items():
            try:
                await self._send_batch(any_llm_key, events)
            except Exception as exc:
                logger.error(f"Failed to send usage event batch: {exc}")

    async def _send_batch(self, any_llm_key: str, events: Sequence[UsageEventBatch]) -> None:
        """Send a batch of events to the platform API.

        Args:
            any_llm_key: The API key for authentication.
            events: List of events to send.
        """
        if not events:
            return

        try:
            access_token = await self.platform_client._aensure_valid_token(any_llm_key)

            payloads = [event.to_payload() for event in events if event.to_payload()]
            if not payloads:
                return

            response = await self.http_client.post(
                f"{self.platform_client.any_llm_platform_url}/usage-events/bulk",
                json={"events": payloads},
                headers={"Authorization": f"Bearer {access_token}"},
            )
            response.raise_for_status()

            logger.debug(f"Successfully sent batch of {len(payloads)} usage events")

        except Exception as exc:
            logger.error(f"Failed to send usage event batch: {exc}")
            raise

    async def flush(self) -> None:
        """Manually flush all pending events."""
        async with self._lock:
            await self._flush_internal()

    async def shutdown(self) -> None:
        """Shutdown the queue and flush remaining events."""
        self._shutdown = True

        # Cancel background task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self.flush()


# Global singleton instance
_global_batch_queue: UsageEventBatchQueue | None = None
_shutdown_registered = False


def _cleanup_on_exit() -> None:
    """Cleanup function called on process exit."""
    global _global_batch_queue  # noqa: PLW0602

    if _global_batch_queue is not None:
        logger.info("Flushing remaining usage events on process exit...")
        try:
            # Create new event loop if none exists
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run shutdown in the event loop
            loop.run_until_complete(_global_batch_queue.shutdown())
            logger.info("Successfully flushed remaining usage events")
        except Exception as exc:
            logger.error(f"Failed to flush usage events on exit: {exc}")


def get_global_batch_queue(
    platform_client: AnyLLMPlatformClient,
    http_client: httpx.AsyncClient,
    batch_size: int = 50,
    flush_interval: float = 5.0,
) -> UsageEventBatchQueue:
    """Get or create the global batch queue singleton.

    Automatically registers an atexit handler to flush remaining events
    on process exit.

    Args:
        platform_client: Platform client for authentication.
        http_client: HTTP client for making requests.
        batch_size: Maximum number of events per batch.
        flush_interval: Maximum seconds to wait before flushing.

    Returns:
        The global batch queue instance.
    """
    global _global_batch_queue, _shutdown_registered  # noqa: PLW0603

    if _global_batch_queue is None:
        _global_batch_queue = UsageEventBatchQueue(
            platform_client=platform_client,
            http_client=http_client,
            batch_size=batch_size,
            flush_interval=flush_interval,
        )

        # Register cleanup handler once
        if not _shutdown_registered:
            atexit.register(_cleanup_on_exit)
            _shutdown_registered = True
            logger.debug("Registered automatic cleanup handler for usage event batching")

    return _global_batch_queue


async def shutdown_global_batch_queue() -> None:
    """Shutdown and flush the global batch queue.

    This function can be called explicitly for graceful shutdown.
    Note: An atexit handler is automatically registered to flush
    events on process exit, so manual calls are optional but recommended
    for cleaner shutdown in long-running applications.
    """
    global _global_batch_queue  # noqa: PLW0603

    if _global_batch_queue is not None:
        await _global_batch_queue.shutdown()
        _global_batch_queue = None
        logger.debug("Global batch queue shutdown complete")
