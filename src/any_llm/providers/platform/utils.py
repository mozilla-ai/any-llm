from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any

import httpx  # noqa: TC002
from any_llm_platform_client import (
    AnyLLMPlatformClient,  # noqa: TC002
)

from .batch_queue import UsageEventBatchQueue
from any_llm import __version__

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion

ANY_LLM_PLATFORM_URL = os.getenv("ANY_LLM_PLATFORM_URL", "https://platform-api.any-llm.ai")
API_V1_STR = "/api/v1"
ANY_LLM_PLATFORM_API_URL = f"{ANY_LLM_PLATFORM_URL}{API_V1_STR}"

__all__ = [
    "UsageEventBatchQueue",
    "build_usage_event_payload",
    "post_completion_usage_event",
    "queue_completion_usage_event",
]


def build_usage_event_payload(
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
) -> dict[str, Any]:
    """Build a usage event payload for the Any LLM Platform API.

    This function defines the payload format expected by the platform's usage events endpoint.
    It's a public API that can be used by any code that needs to construct usage event payloads
    for the platform.

    Args:
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

    Returns:
        Dictionary containing the usage event payload, or empty dict if no usage data.
    """
    if completion.usage is None:
        return {}

    data: dict[str, Any] = {
        "input_tokens": str(completion.usage.prompt_tokens),
        "output_tokens": str(completion.usage.completion_tokens),
    }

    performance: dict[str, float | int] = {}
    if time_to_first_token_ms is not None:
        performance["time_to_first_token_ms"] = time_to_first_token_ms
    if time_to_last_token_ms is not None:
        performance["time_to_last_token_ms"] = time_to_last_token_ms
    if total_duration_ms is not None:
        performance["total_duration_ms"] = total_duration_ms
    if tokens_per_second is not None:
        performance["tokens_per_second"] = tokens_per_second
    if chunks_received is not None:
        performance["chunks_received"] = chunks_received
    if avg_chunk_size is not None:
        performance["avg_chunk_size"] = avg_chunk_size
    if inter_chunk_latency_variance_ms is not None:
        performance["inter_chunk_latency_variance_ms"] = inter_chunk_latency_variance_ms

    if performance:
        data["performance"] = performance

    payload = {
        "id": str(uuid.uuid4()),
        "provider_key_id": provider_key_id,
        "provider": provider,
        "model": completion.model,
        "data": data,
    }
    if client_name:
        payload["client_name"] = client_name

    return payload


async def queue_completion_usage_event(
    batch_queue: UsageEventBatchQueue,
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
    """Queues completion usage events for batch sending (recommended).

    This function adds the event to a batch queue that automatically flushes
    either when the batch size is reached or after the flush interval expires.
    This significantly reduces HTTP overhead compared to sending each event individually.

    Uses JWT Bearer token authentication to authenticate with the platform API.

    Args:
        batch_queue: The UsageEventBatchQueue instance to use for batching.
        any_llm_key: The Any LLM platform key, tied to a specific project.
        provider: The name of the LLM provider.
        completion: The LLM response.
        provider_key_id: The unique identifier for the provider key.
        client_name: Optional name of the client for per-client usage tracking.
        time_to_first_token_ms: Time to first token in milliseconds (streaming only).
        time_to_last_token_ms: Time to last token in milliseconds (streaming only).
        total_duration_ms: Total request duration in milliseconds.
        tokens_per_second: Average token generation throughput.
        chunks_received: Number of chunks received (streaming only).
        avg_chunk_size: Average tokens per chunk (streaming only).
        inter_chunk_latency_variance_ms: Inter-chunk latency variance (streaming only).
    """
    await batch_queue.enqueue(
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


async def post_completion_usage_event(
    platform_client: AnyLLMPlatformClient,
    client: httpx.AsyncClient,
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
    """Posts completion usage events immediately (not recommended for high throughput).

    DEPRECATED: Use `queue_completion_usage_event` instead for better performance.
    This function sends each event individually which creates significant HTTP overhead.

    Uses JWT Bearer token authentication to authenticate with the platform API.

    Args:
        platform_client: The AnyLLMPlatformClient instance to use for authentication.
        client: An httpx client to perform post request.
        any_llm_key: The Any LLM platform key, tied to a specific project.
        provider: The name of the LLM provider.
        completion: The LLM response.
        provider_key_id: The unique identifier for the provider key.
        client_name: Optional name of the client for per-client usage tracking.
        time_to_first_token_ms: Time to first token in milliseconds (streaming only).
        time_to_last_token_ms: Time to last token in milliseconds (streaming only).
        total_duration_ms: Total request duration in milliseconds.
        tokens_per_second: Average token generation throughput.
        chunks_received: Number of chunks received (streaming only).
        avg_chunk_size: Average tokens per chunk (streaming only).
        inter_chunk_latency_variance_ms: Inter-chunk latency variance (streaming only).
    """
    access_token = await platform_client._aensure_valid_token(any_llm_key)

    if completion.usage is None:
        return

    event_id = str(uuid.uuid4())

    data: dict[str, Any] = {
        "input_tokens": str(completion.usage.prompt_tokens),
        "output_tokens": str(completion.usage.completion_tokens),
    }

    performance: dict[str, float | int] = {}
    if time_to_first_token_ms is not None:
        performance["time_to_first_token_ms"] = time_to_first_token_ms
    if time_to_last_token_ms is not None:
        performance["time_to_last_token_ms"] = time_to_last_token_ms
    if total_duration_ms is not None:
        performance["total_duration_ms"] = total_duration_ms
    if tokens_per_second is not None:
        performance["tokens_per_second"] = tokens_per_second
    if chunks_received is not None:
        performance["chunks_received"] = chunks_received
    if avg_chunk_size is not None:
        performance["avg_chunk_size"] = avg_chunk_size
    if inter_chunk_latency_variance_ms is not None:
        performance["inter_chunk_latency_variance_ms"] = inter_chunk_latency_variance_ms

    if performance:
        data["performance"] = performance

    payload = {
        "provider_key_id": provider_key_id,
        "provider": provider,
        "model": completion.model,
        "data": data,
        "id": event_id,
    }
    if client_name:
        payload["client_name"] = client_name

    response = await client.post(
        f"{ANY_LLM_PLATFORM_API_URL}/usage-events/",
        json=payload,
        headers={
            "Authorization": f"Bearer {access_token}",
            "User-Agent": f"python-any-llm/{__version__}",
        },
    )
    response.raise_for_status()
