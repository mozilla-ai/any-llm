"""Unit tests for gateway Prometheus metrics — no database required."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.metrics import (
    REGISTRY,
    MetricsMiddleware,
    metrics_endpoint,
    record_auth_failure,
    record_budget_exceeded,
    record_cost,
    record_rate_limit_hit,
    record_tokens,
)


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    """Read a metric sample value from the registry, returning 0.0 if not found."""
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def test_record_tokens_increments_counters() -> None:
    input_labels = {"provider": "test-prov", "model": "test-model", "type": "input"}
    output_labels = {"provider": "test-prov", "model": "test-model", "type": "output"}
    before_in = _sample("gateway_tokens_total", input_labels)
    before_out = _sample("gateway_tokens_total", output_labels)

    record_tokens("test-prov", "test-model", 100, 50)

    assert _sample("gateway_tokens_total", input_labels) - before_in == 100.0
    assert _sample("gateway_tokens_total", output_labels) - before_out == 50.0


def test_record_tokens_skips_zero_values() -> None:
    labels_in = {"provider": "zero-prov", "model": "zero-model", "type": "input"}
    labels_out = {"provider": "zero-prov", "model": "zero-model", "type": "output"}
    before_in = _sample("gateway_tokens_total", labels_in)
    before_out = _sample("gateway_tokens_total", labels_out)

    record_tokens("zero-prov", "zero-model", 0, 0)

    assert _sample("gateway_tokens_total", labels_in) == before_in
    assert _sample("gateway_tokens_total", labels_out) == before_out


def test_record_cost_observes_histogram() -> None:
    labels = {"provider": "cost-prov", "model": "cost-model"}
    before_count = _sample("gateway_request_cost_dollars_count", labels)

    record_cost("cost-prov", "cost-model", 1.23)

    assert _sample("gateway_request_cost_dollars_count", labels) - before_count == 1.0
    assert _sample("gateway_request_cost_dollars_sum", labels) >= 1.23


def test_record_rate_limit_hit_increments_counter() -> None:
    labels = {"user_id": "rl-unit-user"}
    before = _sample("gateway_rate_limit_hits_total", labels)

    record_rate_limit_hit("rl-unit-user")

    assert _sample("gateway_rate_limit_hits_total", labels) - before == 1.0


def test_record_budget_exceeded_increments_counter() -> None:
    labels = {"user_id": "budget-unit-user"}
    before = _sample("gateway_budget_exceeded_total", labels)

    record_budget_exceeded("budget-unit-user")

    assert _sample("gateway_budget_exceeded_total", labels) - before == 1.0


def test_record_auth_failure_increments_counter() -> None:
    labels = {"reason": "unit-test-reason"}
    before = _sample("gateway_auth_failures_total", labels)

    record_auth_failure("unit-test-reason")

    assert _sample("gateway_auth_failures_total", labels) - before == 1.0


def test_config_enable_metrics_defaults_to_false() -> None:
    config = GatewayConfig()
    assert config.enable_metrics is False


def test_config_enable_metrics_accepted() -> None:
    config = GatewayConfig(enable_metrics=True)
    assert config.enable_metrics is True


def _make_test_app(*, enable_metrics: bool = True) -> FastAPI:
    """Build a minimal FastAPI app with /ok and optionally the metrics middleware + endpoint."""
    app = FastAPI()

    @app.get("/ok")
    async def ok() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/error")
    async def error() -> None:
        raise HTTPException(status_code=503, detail="boom")

    if enable_metrics:
        app.add_middleware(MetricsMiddleware)
        app.add_route("/metrics", metrics_endpoint, methods=["GET"])

    return app


def test_middleware_increments_request_counter() -> None:
    app = _make_test_app()
    client = TestClient(app)
    labels = {"method": "GET", "endpoint": "/ok", "status": "200"}
    before = _sample("gateway_requests_total", labels)

    client.get("/ok")

    assert _sample("gateway_requests_total", labels) - before == 1.0


def test_middleware_records_duration() -> None:
    app = _make_test_app()
    client = TestClient(app)
    labels = {"method": "GET", "endpoint": "/ok"}
    before = _sample("gateway_request_duration_seconds_count", labels)

    client.get("/ok")

    assert _sample("gateway_request_duration_seconds_count", labels) - before == 1.0
    assert _sample("gateway_request_duration_seconds_sum", labels) > 0


def test_middleware_tracks_error_status_codes() -> None:
    app = _make_test_app()
    client = TestClient(app, raise_server_exceptions=False)
    labels = {"method": "GET", "endpoint": "/error", "status": "503"}
    before = _sample("gateway_requests_total", labels)

    client.get("/error")

    assert _sample("gateway_requests_total", labels) - before == 1.0


def test_middleware_skips_metrics_endpoint() -> None:
    app = _make_test_app()
    client = TestClient(app)
    labels = {"method": "GET", "endpoint": "/metrics", "status": "200"}
    before = _sample("gateway_requests_total", labels)

    client.get("/metrics")
    client.get("/metrics")

    assert _sample("gateway_requests_total", labels) == before


def test_metrics_endpoint_returns_prometheus_format() -> None:
    app = _make_test_app()
    client = TestClient(app)
    resp = client.get("/metrics")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "gateway_requests" in resp.text
    assert "gateway_active_requests" in resp.text


def test_metrics_endpoint_not_present_when_disabled() -> None:
    app = _make_test_app(enable_metrics=False)
    client = TestClient(app)
    resp = client.get("/metrics")

    assert resp.status_code in (404, 405)


def test_active_requests_returns_to_zero() -> None:
    """After a request completes, active_requests gauge should be back to its prior value."""
    app = _make_test_app()
    client = TestClient(app)
    before = _sample("gateway_active_requests")

    client.get("/ok")

    assert _sample("gateway_active_requests") == before


def test_rate_limiter_records_metric_on_429() -> None:
    """RateLimiter.check() records a metric before raising 429."""
    from any_llm.gateway.rate_limit import RateLimiter

    limiter = RateLimiter(rpm=1)
    limiter.check("metric-rl-user")

    labels = {"user_id": "metric-rl-user"}
    before = _sample("gateway_rate_limit_hits_total", labels)

    with pytest.raises(HTTPException) as exc_info:
        limiter.check("metric-rl-user")

    assert exc_info.value.status_code == 429
    assert _sample("gateway_rate_limit_hits_total", labels) - before == 1.0
