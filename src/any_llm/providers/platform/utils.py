from __future__ import annotations

import atexit
import os
import re
import threading
from typing import TYPE_CHECKING, cast

import httpx  # noqa: TC002
from any_llm_platform_client import (
    AnyLLMPlatformClient,  # noqa: TC002
)
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind
from typing_extensions import override

from any_llm import __version__
from any_llm.logging import logger

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.trace import Span

    from any_llm.types.completion import ChatCompletion

ANY_LLM_PLATFORM_API_URL = os.getenv("ANY_LLM_PLATFORM_URL", "https://platform-api.any-llm.ai/api/v1").rstrip("/")
TRACE_API_PATH = "/v1/traces"

_trace_base_url = ANY_LLM_PLATFORM_API_URL
_trace_base_url = _trace_base_url.removesuffix("/api/v1")
ANY_LLM_PLATFORM_TRACE_URL = f"{_trace_base_url}{TRACE_API_PATH}"


def _is_localhost_url(url: str) -> bool:
    """Return True when *url* points to a loopback address (localhost / 127.0.0.1)."""
    from urllib.parse import urlparse

    hostname = urlparse(url).hostname or ""
    return hostname in ("localhost", "127.0.0.1", "::1")


def _require_secure_trace_endpoint() -> str:
    if ANY_LLM_PLATFORM_TRACE_URL.startswith("https://"):
        return ANY_LLM_PLATFORM_TRACE_URL
    if _is_localhost_url(ANY_LLM_PLATFORM_TRACE_URL):
        return ANY_LLM_PLATFORM_TRACE_URL
    msg = f"ANY_LLM_PLATFORM_TRACE_URL must use HTTPS: {ANY_LLM_PLATFORM_TRACE_URL}"
    raise ValueError(msg)


# Module-level cache: access_token -> TracerProvider
_providers: dict[str, TracerProvider] = {}
_providers_lock = threading.Lock()

# Module-level cache: access_token -> BatchSpanProcessor used for forwarding
_forward_processors: dict[str, BatchSpanProcessor] = {}
_forward_processors_lock = threading.Lock()

# Trace ids currently in platform-provider call scope.
# We only forward spans for trace_ids in this map.
_active_trace_exports: dict[int, tuple[str, int]] = {}
_active_trace_exports_lock = threading.Lock()

_forwarding_processor_holder: dict[str, PlatformScopedForwardingSpanProcessor] = {}
_forwarding_processor_lock = threading.Lock()

_CONTENT_ATTRIBUTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|[._])prompt($|[._])", re.IGNORECASE),
    re.compile(r"(^|[._])message($|[._])", re.IGNORECASE),
    re.compile(r"(^|[._])messages($|[._])", re.IGNORECASE),
    re.compile(r"(^|[._])content($|[._])", re.IGNORECASE),
    re.compile(r"(^|[._])payload($|[._])", re.IGNORECASE),
    re.compile(r"(^|[._])body($|[._])", re.IGNORECASE),
    re.compile(r"(^|[._])response($|[._])", re.IGNORECASE),
    re.compile(r"(^|[._])completion($|[._])", re.IGNORECASE),
    re.compile(r"(^|[._])input($|[._])", re.IGNORECASE),
    re.compile(r"(^|[._])output($|[._])", re.IGNORECASE),
)

_SAFE_ATTRIBUTE_TOKENS: tuple[str, ...] = (
    "token",
    "model",
    "usage",
    "latency",
    "duration",
    "status",
    "provider",
    "session",
    "client",
    "trace",
    "span",
    "id",
    "cost",
    "error",
)


def _is_content_attribute_key(key: str) -> bool:
    key_l = key.lower()
    if any(tok in key_l for tok in _SAFE_ATTRIBUTE_TOKENS):
        return False
    return any(pattern.search(key_l) is not None for pattern in _CONTENT_ATTRIBUTE_PATTERNS)


def _sanitize_attribute_mapping(attributes: object | None) -> None:
    if attributes is None:
        return
    if not hasattr(attributes, "keys"):
        return

    mutable_attributes = cast("MutableMapping[str, object]", attributes)
    supports_delete = hasattr(attributes, "__delitem__")
    supports_set = hasattr(attributes, "__setitem__")

    keys_to_remove = [key for key in mutable_attributes if _is_content_attribute_key(key)]
    for key in keys_to_remove:
        if supports_delete:
            try:
                del mutable_attributes[key]
                continue
            except (KeyError, TypeError):
                pass
        if supports_set:
            try:
                mutable_attributes[key] = "[redacted]"
            except (KeyError, TypeError):
                continue


def _sanitized_attributes_copy(attributes: object | None) -> dict[str, object] | None:
    if attributes is None or not hasattr(attributes, "items"):
        return None
    copied = {str(key): value for key, value in cast("MutableMapping[str, object]", attributes).items()}
    keys_to_remove = [key for key in copied if _is_content_attribute_key(key)]
    for key in keys_to_remove:
        copied.pop(key, None)
    return copied


class _SanitizedEventProxy:
    def __init__(self, event: object):
        self._event = event
        self.attributes = _sanitized_attributes_copy(getattr(event, "attributes", None))

    def __getattr__(self, item: str) -> object:
        return getattr(self._event, item)


class _SanitizedReadableSpanProxy:
    def __init__(self, span: ReadableSpan):
        self._span = span
        self.attributes = _sanitized_attributes_copy(span.attributes)
        self.events = tuple(_SanitizedEventProxy(event) for event in span.events)

    def __getattr__(self, item: str) -> object:
        return getattr(self._span, item)


def _get_or_create_forward_processor(access_token: str) -> BatchSpanProcessor:
    with _forward_processors_lock:
        if access_token in _forward_processors:
            return _forward_processors[access_token]

        exporter = OTLPSpanExporter(
            endpoint=_require_secure_trace_endpoint(),
            headers={
                "Authorization": f"Bearer {access_token}",
                "User-Agent": f"python-any-llm/{__version__}",
            },
        )
        processor = BatchSpanProcessor(exporter)
        _forward_processors[access_token] = processor
        return processor


class PlatformScopedForwardingSpanProcessor(SpanProcessor):
    @override
    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        return

    @override
    def on_end(self, span: ReadableSpan) -> None:
        span_context = span.context
        if span_context is None:
            return

        with _active_trace_exports_lock:
            trace_data = _active_trace_exports.get(span_context.trace_id)
        if trace_data is None:
            return

        access_token, _ = trace_data
        sanitized_span = cast("ReadableSpan", _SanitizedReadableSpanProxy(span))
        processor = _get_or_create_forward_processor(access_token)
        processor.on_end(sanitized_span)

    @override
    def shutdown(self) -> None:
        with _forward_processors_lock:
            for processor in _forward_processors.values():
                processor.shutdown()  # type: ignore[no-untyped-call]
            _forward_processors.clear()

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        ok = True
        with _forward_processors_lock:
            for processor in _forward_processors.values():
                ok = processor.force_flush(timeout_millis=timeout_millis) and ok
        return ok


def _ensure_forwarding_processor_installed() -> None:
    with _forwarding_processor_lock:
        if "processor" in _forwarding_processor_holder:
            return

        provider = trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            return

        processor = PlatformScopedForwardingSpanProcessor()
        provider.add_span_processor(processor)
        _forwarding_processor_holder["processor"] = processor


def activate_trace_export(trace_id: int, access_token: str) -> None:
    _ensure_forwarding_processor_installed()
    with _active_trace_exports_lock:
        existing = _active_trace_exports.get(trace_id)
        if existing is None:
            _active_trace_exports[trace_id] = (access_token, 1)
            return
        if existing[0] != access_token:
            logger.warning(
                "Conflicting access tokens for trace_id=%s; skipping scoped export activation to avoid cross-token leakage",
                trace_id,
            )
            return
        _active_trace_exports[trace_id] = (access_token, existing[1] + 1)


def deactivate_trace_export(trace_id: int) -> None:
    with _active_trace_exports_lock:
        existing = _active_trace_exports.get(trace_id)
        if existing is None:
            return
        access_token, count = existing
        if count <= 1:
            _active_trace_exports.pop(trace_id, None)
        else:
            _active_trace_exports[trace_id] = (access_token, count - 1)


def _get_or_create_tracer_provider(access_token: str) -> TracerProvider:
    """Get or create a TracerProvider for the given access token.

    Providers are cached by token and reused across requests.
    When a token expires and a new one is issued, a new provider is created.
    """
    with _providers_lock:
        if access_token in _providers:
            return _providers[access_token]

        provider = TracerProvider(resource=Resource.create({"service.name": "any-llm"}))
        exporter = OTLPSpanExporter(
            endpoint=_require_secure_trace_endpoint(),
            headers={
                "Authorization": f"Bearer {access_token}",
                "User-Agent": f"python-any-llm/{__version__}",
            },
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        _providers[access_token] = provider
        return provider


def shutdown_telemetry() -> None:
    """Shutdown all cached tracer providers.

    Called automatically at process exit via atexit, but can also be called
    manually to ensure all pending spans are flushed before shutdown.
    """
    with _providers_lock:
        for provider in _providers.values():
            provider.shutdown()
        _providers.clear()

    with _forward_processors_lock:
        for processor in _forward_processors.values():
            processor.shutdown()  # type: ignore[no-untyped-call]
        _forward_processors.clear()

    with _active_trace_exports_lock:
        _active_trace_exports.clear()


atexit.register(shutdown_telemetry)


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
    user_session_label: str | None = None,
    include_session_label_attribute: bool = True,
    conversation_id: str | None = None,
    access_token: str | None = None,
    existing_span: Span | None = None,
) -> None:
    """Export an OTLP trace span for an LLM completion.

    Uses JWT Bearer token authentication to authenticate with the platform API.
    Prompts and responses are never included in trace attributes.
    """
    token = access_token or await platform_client._aensure_valid_token(any_llm_key)

    if existing_span is not None:
        span = existing_span
    else:
        provider_instance = _get_or_create_tracer_provider(token)
        tracer = provider_instance.get_tracer("any-llm", __version__)

        # Context resolution priority:
        # 1. Caller's active OTel span (explicit instrumentation) takes precedence.
        # 2. No active context -> create an independent root span.
        current_ctx = otel_context.get_current()
        caller_span = trace.get_current_span(current_ctx)
        caller_has_active_span = caller_span.get_span_context().is_valid

        parent_ctx = current_ctx if caller_has_active_span else None

        if parent_ctx is not None:
            span = tracer.start_span("llm.request", kind=SpanKind.CLIENT, start_time=start_time_ns, context=parent_ctx)
        else:
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
    if session_label is not None and include_session_label_attribute:
        span.set_attribute("anyllm.session_label", session_label)
    if user_session_label is not None:
        span.set_attribute("anyllm.user_session_label", user_session_label)

    span.end(end_time=end_time_ns)


async def export_responses_trace(
    platform_client: AnyLLMPlatformClient,
    client: httpx.AsyncClient,
    any_llm_key: str,
    provider: str,
    request_model: str,
    response_model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    start_time_ns: int,
    end_time_ns: int,
    client_name: str | None = None,
    session_label: str | None = None,
    user_session_label: str | None = None,
    conversation_id: str | None = None,
    access_token: str | None = None,
    existing_span: Span | None = None,
) -> None:
    """Export an OTLP trace span for a Responses API call.

    Uses JWT Bearer token authentication to authenticate with the platform API.
    Prompts and responses are never included in trace attributes.
    """
    token = access_token or await platform_client._aensure_valid_token(any_llm_key)

    if existing_span is not None:
        span = existing_span
    else:
        provider_instance = _get_or_create_tracer_provider(token)
        tracer = provider_instance.get_tracer("any-llm", __version__)

        current_ctx = otel_context.get_current()
        caller_span = trace.get_current_span(current_ctx)
        caller_has_active_span = caller_span.get_span_context().is_valid

        parent_ctx = current_ctx if caller_has_active_span else None

        if parent_ctx is not None:
            span = tracer.start_span("llm.request", kind=SpanKind.CLIENT, start_time=start_time_ns, context=parent_ctx)
        else:
            span = tracer.start_span("llm.request", kind=SpanKind.CLIENT, start_time=start_time_ns)

    span.set_attribute("gen_ai.provider.name", provider)
    span.set_attribute("gen_ai.request.model", request_model)

    if response_model is not None:
        span.set_attribute("gen_ai.response.model", response_model)
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)

    if conversation_id is not None:
        span.set_attribute("gen_ai.conversation.id", conversation_id)
    if client_name is not None:
        span.set_attribute("anyllm.client_name", client_name)
    if session_label is not None:
        span.set_attribute("anyllm.session_label", session_label)
    if user_session_label is not None:
        span.set_attribute("anyllm.user_session_label", user_session_label)

    span.end(end_time=end_time_ns)
