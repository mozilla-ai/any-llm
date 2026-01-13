from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING, Any

import httpx  # noqa: TC002
from any_llm_platform_client import (
    AnyLLMPlatformClient,  # noqa: TC002
)

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion


ANY_LLM_PLATFORM_URL = os.getenv("ANY_LLM_PLATFORM_URL", "https://platform-api.any-llm.ai")
API_V1_STR = "/api/v1"
ANY_LLM_PLATFORM_API_URL = f"{ANY_LLM_PLATFORM_URL}{API_V1_STR}"


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
    """Posts completion usage events.

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
        headers={"Authorization": f"Bearer {access_token}"},
    )
    response.raise_for_status()
