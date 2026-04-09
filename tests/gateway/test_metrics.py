"""Tests for Prometheus metrics instrumentation."""

from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from any_llm.gateway.core.config import API_KEY_HEADER, GatewayConfig
from any_llm.gateway.core.database import _make_async_engine
from any_llm.gateway.db import get_db
from any_llm.gateway.main import create_app
from any_llm.gateway.metrics import REGISTRY
from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage
from tests.gateway.conftest import _drop_all_sync, _run_alembic_migrations


def _sample(name: str, labels: dict[str, str]) -> float:
    """Read a metric sample value from the registry, returning 0.0 if not found."""
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _make_metrics_client(
    postgres_url: str,
    *,
    enable_metrics: bool = True,
    rate_limit_rpm: int | None = None,
) -> Generator[TestClient]:
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        enable_metrics=enable_metrics,
        rate_limit_rpm=rate_limit_rpm,
    )
    _run_alembic_migrations(postgres_url)
    engine = _make_async_engine(postgres_url)
    session_factory = async_sessionmaker(engine, autocommit=False, autoflush=False, expire_on_commit=False)
    app = create_app(config)

    async def override_get_db() -> AsyncGenerator[AsyncSession]:
        async with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        _drop_all_sync(postgres_url)


@pytest.fixture
def metrics_client(postgres_url: str) -> Generator[TestClient]:
    """TestClient with metrics enabled."""
    yield from _make_metrics_client(postgres_url)


@pytest.fixture
def no_metrics_client(postgres_url: str) -> Generator[TestClient]:
    """TestClient with metrics disabled."""
    yield from _make_metrics_client(postgres_url, enable_metrics=False)


@pytest.fixture
def metrics_rate_limit_client(postgres_url: str) -> Generator[TestClient]:
    """TestClient with metrics and rate limiting enabled."""
    yield from _make_metrics_client(postgres_url, rate_limit_rpm=2)


def _create_user(client: TestClient, user_id: str = "metrics-test-user") -> str:
    header = {API_KEY_HEADER: "Bearer test-master-key"}
    resp = client.post("/v1/users", json={"user_id": user_id, "alias": "Metrics"}, headers=header)
    assert resp.status_code == 200
    result: str = resp.json()["user_id"]
    return result


def _chat_request(client: TestClient, user_id: str) -> Any:
    header = {API_KEY_HEADER: "Bearer test-master-key"}
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "user": user_id,
        },
        headers=header,
    )


def test_metrics_endpoint_returns_prometheus_format(metrics_client: TestClient) -> None:
    resp = metrics_client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    assert "gateway_requests_total" in body
    assert "gateway_active_requests" in body


def test_metrics_endpoint_not_available_when_disabled(no_metrics_client: TestClient) -> None:
    resp = no_metrics_client.get("/metrics")
    assert resp.status_code == 404


def test_request_counter_increments_on_health_check(metrics_client: TestClient) -> None:
    labels = {"method": "GET", "endpoint": "/health", "status": "200"}
    before = _sample("gateway_requests_total", labels)

    metrics_client.get("/health")

    after = _sample("gateway_requests_total", labels)
    assert after - before == 1.0


def test_request_duration_recorded(metrics_client: TestClient) -> None:
    labels = {"method": "GET", "endpoint": "/health"}
    before = _sample("gateway_request_duration_seconds_count", labels)

    metrics_client.get("/health")

    after = _sample("gateway_request_duration_seconds_count", labels)
    assert after - before == 1.0


def test_token_metrics_recorded(metrics_client: TestClient) -> None:
    user_id = _create_user(metrics_client)

    mock_response = ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model="gpt-4o-mini",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    input_labels = {"provider": "openai", "model": "gpt-4o-mini", "type": "input"}
    output_labels = {"provider": "openai", "model": "gpt-4o-mini", "type": "output"}
    before_input = _sample("gateway_tokens_total", input_labels)
    before_output = _sample("gateway_tokens_total", output_labels)

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        return mock_response

    with patch("any_llm.gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = _chat_request(metrics_client, user_id)

    assert resp.status_code == 200
    assert _sample("gateway_tokens_total", input_labels) - before_input == 10.0
    assert _sample("gateway_tokens_total", output_labels) - before_output == 5.0


def test_cost_metric_recorded_with_pricing(metrics_client: TestClient) -> None:
    user_id = _create_user(metrics_client, user_id="cost-user")

    # Set up pricing
    header = {API_KEY_HEADER: "Bearer test-master-key"}
    metrics_client.post(
        "/v1/pricing",
        json={
            "model_key": "openai:gpt-4o-mini",
            "input_price_per_million": 1.0,
            "output_price_per_million": 2.0,
        },
        headers=header,
    )

    mock_response = ChatCompletion(
        id="chatcmpl-test",
        object="chat.completion",
        created=1700000000,
        model="gpt-4o-mini",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hello"),
                finish_reason="stop",
            )
        ],
        usage=CompletionUsage(prompt_tokens=1_000_000, completion_tokens=500_000, total_tokens=1_500_000),
    )

    cost_labels = {"provider": "openai", "model": "gpt-4o-mini"}
    before = _sample("gateway_request_cost_dollars_count", cost_labels)

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        return mock_response

    with patch("any_llm.gateway.api.routes.chat.acompletion", new=mock_acompletion):
        resp = _chat_request(metrics_client, user_id)

    assert resp.status_code == 200
    assert _sample("gateway_request_cost_dollars_count", cost_labels) - before == 1.0
    cost_sum_after = _sample("gateway_request_cost_dollars_sum", cost_labels)
    # Expected cost: (1M / 1M) * 1.0 + (500K / 1M) * 2.0 = 1.0 + 1.0 = 2.0
    assert cost_sum_after > 0


def test_rate_limit_hit_metric(metrics_rate_limit_client: TestClient) -> None:
    user_id = _create_user(metrics_rate_limit_client, user_id="rl-metric-user")

    labels = {"user_id": user_id}
    before = _sample("gateway_rate_limit_hits_total", labels)

    async def mock_acompletion(**kwargs: Any) -> None:
        msg = "short-circuit"
        raise RuntimeError(msg)

    with patch("any_llm.gateway.api.routes.chat.acompletion", new=mock_acompletion):
        # Exhaust rate limit (rpm=2)
        for _ in range(2):
            _chat_request(metrics_rate_limit_client, user_id)
        # This one should be rate-limited
        resp = _chat_request(metrics_rate_limit_client, user_id)

    assert resp.status_code == 429
    assert _sample("gateway_rate_limit_hits_total", labels) - before >= 1.0


def test_budget_exceeded_metric(metrics_client: TestClient) -> None:
    header = {API_KEY_HEADER: "Bearer test-master-key"}

    # Create budget with 0 limit
    budget_resp = metrics_client.post(
        "/v1/budgets",
        json={"max_budget": 0.0},
        headers=header,
    )
    assert budget_resp.status_code == 200
    budget_id = budget_resp.json()["budget_id"]

    # Create user with that budget
    user_resp = metrics_client.post(
        "/v1/users",
        json={"user_id": "budget-user", "alias": "Budget", "budget_id": budget_id},
        headers=header,
    )
    assert user_resp.status_code == 200

    labels = {"user_id": "budget-user"}
    before = _sample("gateway_budget_exceeded_total", labels)

    resp = _chat_request(metrics_client, "budget-user")
    assert resp.status_code == 403

    assert _sample("gateway_budget_exceeded_total", labels) - before == 1.0


def test_auth_failure_metric_missing_credentials(metrics_client: TestClient) -> None:
    labels = {"reason": "missing_credentials"}
    before = _sample("gateway_auth_failures_total", labels)

    resp = metrics_client.post(
        "/v1/chat/completions",
        json={"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401

    assert _sample("gateway_auth_failures_total", labels) - before == 1.0


def test_auth_failure_metric_invalid_key(metrics_client: TestClient) -> None:
    # Use a valid-format key (gw- prefix, 50+ chars) that doesn't exist in the DB
    fake_key = "gw-" + "a" * 48
    labels = {"reason": "invalid_key"}
    before = _sample("gateway_auth_failures_total", labels)

    resp = metrics_client.post(
        "/v1/chat/completions",
        json={"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        headers={API_KEY_HEADER: f"Bearer {fake_key}"},
    )
    assert resp.status_code == 401

    assert _sample("gateway_auth_failures_total", labels) - before == 1.0


def test_metrics_not_self_instrumented(metrics_client: TestClient) -> None:
    """The /metrics endpoint should not increment request counters for itself."""
    labels = {"method": "GET", "endpoint": "/metrics", "status": "200"}
    before = _sample("gateway_requests_total", labels)

    metrics_client.get("/metrics")
    metrics_client.get("/metrics")

    after = _sample("gateway_requests_total", labels)
    assert after == before


def test_request_counter_tracks_error_status(metrics_client: TestClient) -> None:
    labels = {"method": "POST", "endpoint": "/v1/chat/completions", "status": "401"}
    before = _sample("gateway_requests_total", labels)

    metrics_client.post(
        "/v1/chat/completions",
        json={"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    )

    after = _sample("gateway_requests_total", labels)
    assert after - before == 1.0
