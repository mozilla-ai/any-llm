from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING, Any

import httpx  # noqa: TC002
from any_llm_platform_client import (
    AnyLLMPlatformClient,  # noqa: TC002
)

from any_llm import __version__

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion

ANY_LLM_PLATFORM_API_URL = os.getenv("ANY_LLM_PLATFORM_URL", "https://platform-api.any-llm.ai/api/v1").rstrip("/")
TRACE_API_PATH = "/v1/traces"

_trace_base_url = ANY_LLM_PLATFORM_API_URL
if _trace_base_url.endswith("/api/v1"):
    _trace_base_url = _trace_base_url[: -len("/api/v1")]
ANY_LLM_PLATFORM_TRACE_URL = f"{_trace_base_url}{TRACE_API_PATH}"


def _generate_trace_ids() -> tuple[str, str]:
    trace_id = base64.b64encode(os.urandom(16)).decode("ascii")
    span_id = base64.b64encode(os.urandom(8)).decode("ascii")
    return trace_id, span_id


def _attribute_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def _append_attribute(attributes: list[dict[str, Any]], key: str, value: Any) -> None:
    if value is None:
        return
    attributes.append({"key": key, "value": _attribute_value(value)})


async def export_completion_trace(
    platform_client: AnyLLMPlatformClient,
    client: httpx.AsyncClient,
    any_llm_key: str,
    provider: str,
    request_model: str,
    completion: ChatCompletion | None,
    start_time_ns: int,
    end_time_ns: int,
    client_name: str | None = None,
    session_label: str | None = None,
    conversation_id: str | None = None,
    time_to_first_token_ms: float | None = None,
    time_to_last_token_ms: float | None = None,
    total_duration_ms: float | None = None,
    tokens_per_second: float | None = None,
    chunks_received: int | None = None,
    avg_chunk_size: float | None = None,
    inter_chunk_latency_variance_ms: float | None = None,
) -> None:
    """Export an OTLP trace span for an LLM completion.

    Uses JWT Bearer token authentication to authenticate with the platform API.
    Prompts and responses are never included in trace attributes.
    """
    access_token = await platform_client._aensure_valid_token(any_llm_key)

    attributes: list[dict[str, Any]] = []
    _append_attribute(attributes, "gen_ai.provider.name", provider)
    _append_attribute(attributes, "gen_ai.request.model", request_model)

    if completion is not None:
        _append_attribute(attributes, "gen_ai.response.model", completion.model)
        usage = completion.usage
        if usage is not None:
            _append_attribute(attributes, "gen_ai.usage.input_tokens", usage.prompt_tokens)
            _append_attribute(attributes, "gen_ai.usage.output_tokens", usage.completion_tokens)

    _append_attribute(attributes, "gen_ai.conversation.id", conversation_id)
    _append_attribute(attributes, "anyllm.client_name", client_name)
    _append_attribute(attributes, "anyllm.session_label", session_label)

    _append_attribute(attributes, "anyllm.performance.time_to_first_token_ms", time_to_first_token_ms)
    _append_attribute(attributes, "anyllm.performance.time_to_last_token_ms", time_to_last_token_ms)
    _append_attribute(attributes, "anyllm.performance.total_duration_ms", total_duration_ms)
    _append_attribute(attributes, "anyllm.performance.tokens_per_second", tokens_per_second)
    _append_attribute(attributes, "anyllm.performance.chunks_received", chunks_received)
    _append_attribute(attributes, "anyllm.performance.avg_chunk_size", avg_chunk_size)
    _append_attribute(
        attributes,
        "anyllm.performance.inter_chunk_latency_variance_ms",
        inter_chunk_latency_variance_ms,
    )

    trace_id, span_id = _generate_trace_ids()

    payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": "any-llm"}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "any-llm", "version": __version__},
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": span_id,
                                "name": "llm.request",
                                "kind": "SPAN_KIND_CLIENT",
                                "startTimeUnixNano": str(start_time_ns),
                                "endTimeUnixNano": str(end_time_ns),
                                "attributes": attributes,
                            }
                        ],
                    }
                ],
            }
        ]
    }

    response = await client.post(
        ANY_LLM_PLATFORM_TRACE_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {access_token}",
            "User-Agent": f"python-any-llm/{__version__}",
            "Content-Type": "application/json",
        },
    )
    response.raise_for_status()
