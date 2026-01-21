"""Batching queue for usage events to reduce HTTP request overhead."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import httpx  # noqa: TC002
from any_llm_platform_client import AnyLLMPlatformClient  # noqa: TC002

from any_llm.logging import logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from any_llm.types.completion import ChatCompletion


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

        self._queue: list[dict[str, Any]] = []
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

        from .utils import build_usage_event_payload

        payload = build_usage_event_payload(
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

        if not payload:
            return

        async with self._lock:
            # Store payload with its any_llm_key for batching
            self._queue.append({"any_llm_key": any_llm_key, "payload": payload})

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
        events_by_key: dict[str, list[dict[str, Any]]] = {}
        for item in self._queue:
            any_llm_key = item["any_llm_key"]
            payload = item["payload"]
            if any_llm_key not in events_by_key:
                events_by_key[any_llm_key] = []
            events_by_key[any_llm_key].append(payload)

        # Clear queue
        self._queue.clear()
        self._last_flush_time = time.monotonic()

        # Send batches for each key
        for any_llm_key, payloads in events_by_key.items():
            try:
                await self._send_batch(any_llm_key, payloads)
            except Exception as exc:
                logger.error(f"Failed to send usage event batch: {exc}")

    async def _send_batch(self, any_llm_key: str, payloads: Sequence[dict[str, Any]]) -> None:
        """Send a batch of events to the platform API.

        Args:
            any_llm_key: The API key for authentication.
            payloads: List of event payloads to send.
        """
        if not payloads:
            return

        try:
            access_token = await self.platform_client._aensure_valid_token(any_llm_key)

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
        """Shutdown the queue and flush remaining events.

        This method ensures all pending events are sent before shutdown,
        even if errors occurred during processing.
        """
        self._shutdown = True

        # Cancel background task
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.error(f"Error cancelling background flush task: {exc}")

        # Final flush - attempt to send all remaining events
        try:
            await self.flush()
        except Exception as exc:
            logger.error(f"Error during final flush on shutdown: {exc}")
            # Log but don't raise - we want shutdown to complete
