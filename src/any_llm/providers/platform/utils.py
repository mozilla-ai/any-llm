from __future__ import annotations

import os
from typing import TYPE_CHECKING

import httpx  # noqa: TC002
from any_llm_platform_client import (
    AnyLLMPlatformClient,  # noqa: TC002
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import SpanKind
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter

from any_llm import __version__

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion

ANY_LLM_PLATFORM_API_URL = os.getenv("ANY_LLM_PLATFORM_URL", "https://platform-api.any-llm.ai/api/v1").rstrip("/")
TRACE_API_PATH = "/v1/traces"

_trace_base_url = ANY_LLM_PLATFORM_API_URL
if _trace_base_url.endswith("/api/v1"):
    _trace_base_url = _trace_base_url[: -len("/api/v1")]
ANY_LLM_PLATFORM_TRACE_URL = f"{_trace_base_url}{TRACE_API_PATH}"


def _build_span_exporter(access_token: str) -> SpanExporter:
    return OTLPSpanExporter(
        endpoint=ANY_LLM_PLATFORM_TRACE_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "User-Agent": f"python-any-llm/{__version__}",
        },
    )


def _build_tracer_provider(access_token: str) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": "any-llm"}))
    provider.add_span_processor(SimpleSpanProcessor(_build_span_exporter(access_token)))
    return provider


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

    provider_instance = _build_tracer_provider(access_token)
    tracer = provider_instance.get_tracer("any-llm", __version__)

    span = tracer.start_span("llm.request", kind=SpanKind.CLIENT, start_time=start_time_ns)

    span.set_attribute("gen_ai.provider.name", provider)
    span.set_attribute("gen_ai.request.model", request_model)

    if completion is not None:
        span.set_attribute("gen_ai.response.model", completion.model)
        usage = completion.usage
        if usage is not None:
            span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)

    if conversation_id is not None:
        span.set_attribute("gen_ai.conversation.id", conversation_id)
    if client_name is not None:
        span.set_attribute("anyllm.client_name", client_name)
    if session_label is not None:
        span.set_attribute("anyllm.session_label", session_label)

    if time_to_first_token_ms is not None:
        span.set_attribute("anyllm.performance.time_to_first_token_ms", time_to_first_token_ms)
    if time_to_last_token_ms is not None:
        span.set_attribute("anyllm.performance.time_to_last_token_ms", time_to_last_token_ms)
    if total_duration_ms is not None:
        span.set_attribute("anyllm.performance.total_duration_ms", total_duration_ms)
    if tokens_per_second is not None:
        span.set_attribute("anyllm.performance.tokens_per_second", tokens_per_second)
    if chunks_received is not None:
        span.set_attribute("anyllm.performance.chunks_received", chunks_received)
    if avg_chunk_size is not None:
        span.set_attribute("anyllm.performance.avg_chunk_size", avg_chunk_size)
    if inter_chunk_latency_variance_ms is not None:
        span.set_attribute(
            "anyllm.performance.inter_chunk_latency_variance_ms",
            inter_chunk_latency_variance_ms,
        )

    span.end(end_time=end_time_ns)
    provider_instance.force_flush()
    provider_instance.shutdown()
