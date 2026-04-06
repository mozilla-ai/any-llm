"""Prometheus metrics for the gateway."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "gateway_requests",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
    registry=REGISTRY,
)

REQUEST_DURATION_SECONDS = Histogram(
    "gateway_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    registry=REGISTRY,
)

ACTIVE_REQUESTS = Gauge(
    "gateway_active_requests",
    "Number of currently in-flight requests",
    registry=REGISTRY,
)

TOKENS = Counter(
    "gateway_tokens",
    "Total number of tokens processed",
    ["provider", "model", "type"],
    registry=REGISTRY,
)

REQUEST_COST_DOLLARS = Histogram(
    "gateway_request_cost_dollars",
    "Request cost in USD",
    ["provider", "model"],
    registry=REGISTRY,
)

RATE_LIMIT_HITS = Counter(
    "gateway_rate_limit_hits",
    "Total number of rate limit hits",
    ["user_id"],
    registry=REGISTRY,
)

BUDGET_EXCEEDED = Counter(
    "gateway_budget_exceeded",
    "Total number of budget exceeded events",
    ["user_id"],
    registry=REGISTRY,
)

AUTH_FAILURES = Counter(
    "gateway_auth_failures",
    "Total number of authentication failures",
    ["reason"],
    registry=REGISTRY,
)


_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


async def metrics_endpoint(request: Request) -> Response:
    """Serve Prometheus metrics."""
    body = generate_latest(REGISTRY)
    return Response(content=body, media_type=_PROMETHEUS_CONTENT_TYPE)


class MetricsMiddleware:
    """ASGI middleware that records request count, duration, and active requests."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path: str = scope["path"]
        if path == "/metrics":
            await self.app(scope, receive, send)
            return

        method: str = scope["method"]
        status_code = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        ACTIVE_REQUESTS.inc()
        start = time.monotonic()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.monotonic() - start
            ACTIVE_REQUESTS.dec()
            REQUESTS.labels(method=method, endpoint=path, status=str(status_code)).inc()
            REQUEST_DURATION_SECONDS.labels(method=method, endpoint=path).observe(duration)


def record_tokens(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Record token usage metrics."""
    if prompt_tokens:
        TOKENS.labels(provider=provider, model=model, type="input").inc(prompt_tokens)
    if completion_tokens:
        TOKENS.labels(provider=provider, model=model, type="output").inc(completion_tokens)


def record_cost(provider: str, model: str, cost: float) -> None:
    """Record request cost."""
    REQUEST_COST_DOLLARS.labels(provider=provider, model=model).observe(cost)


def record_rate_limit_hit(user_id: str) -> None:
    """Record a rate limit hit."""
    RATE_LIMIT_HITS.labels(user_id=user_id).inc()


def record_budget_exceeded(user_id: str) -> None:
    """Record a budget exceeded event."""
    BUDGET_EXCEEDED.labels(user_id=user_id).inc()


def record_auth_failure(reason: str) -> None:
    """Record an authentication failure."""
    AUTH_FAILURES.labels(reason=reason).inc()
