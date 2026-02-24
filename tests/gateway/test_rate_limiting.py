"""Tests for per-user rate limiting."""

from collections.abc import Generator
from typing import Any
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from any_llm.gateway.config import API_KEY_HEADER, GatewayConfig
from any_llm.gateway.db import Base, get_db
from any_llm.gateway.rate_limit import RateLimiter, RateLimitInfo
from any_llm.gateway.server import create_app
from tests.gateway.conftest import _run_alembic_migrations


def test_rate_limiter_allows_under_limit() -> None:
    """Test that requests under the limit are allowed."""
    limiter = RateLimiter(rpm=5)
    for _ in range(5):
        limiter.check("user-1")


def test_rate_limiter_returns_rate_limit_info() -> None:
    """Test that check() returns RateLimitInfo with correct values."""
    limiter = RateLimiter(rpm=5)
    info = limiter.check("user-1")
    assert isinstance(info, RateLimitInfo)
    assert info.limit == 5
    assert info.remaining == 4

    info2 = limiter.check("user-1")
    assert info2.remaining == 3


def test_rate_limiter_reset_is_wall_clock() -> None:
    """Test that reset uses wall-clock time, not monotonic time."""
    limiter = RateLimiter(rpm=5)

    with patch("any_llm.gateway.rate_limit.time") as mock_time:
        mock_time.monotonic.return_value = 1000.0
        mock_time.time.return_value = 1700000000.0
        info = limiter.check("user-1")

    # reset should be based on wall-clock time (~60s from now), not monotonic
    assert info.reset == pytest.approx(1700000060.0, abs=1.0)


def test_rate_limiter_rejects_over_limit() -> None:
    """Test that requests over the limit are rejected with 429."""
    limiter = RateLimiter(rpm=3)
    for _ in range(3):
        limiter.check("user-1")

    with pytest.raises(HTTPException) as exc_info:
        limiter.check("user-1")
    assert exc_info.value.status_code == 429


def test_rate_limiter_429_includes_retry_after() -> None:
    """Test that 429 response includes a Retry-After header."""
    limiter = RateLimiter(rpm=1)
    limiter.check("user-1")

    with pytest.raises(HTTPException) as exc_info:
        limiter.check("user-1")
    assert exc_info.value.headers is not None
    assert "Retry-After" in exc_info.value.headers
    retry_after = int(exc_info.value.headers["Retry-After"])
    assert 1 <= retry_after <= 60


def test_rate_limiter_per_user_isolation() -> None:
    """Test that rate limits are tracked independently per user."""
    limiter = RateLimiter(rpm=2)
    limiter.check("user-1")
    limiter.check("user-1")

    # user-2 should still be allowed
    limiter.check("user-2")
    limiter.check("user-2")

    # user-1 should be blocked, user-2 also blocked
    with pytest.raises(HTTPException):
        limiter.check("user-1")
    with pytest.raises(HTTPException):
        limiter.check("user-2")


def test_rate_limiter_window_expiry() -> None:
    """Test that requests are allowed again after the window expires."""
    limiter = RateLimiter(rpm=2)

    with patch("any_llm.gateway.rate_limit.time") as mock_time:
        mock_time.monotonic.return_value = 1000.0
        mock_time.time.return_value = 1700000000.0
        limiter.check("user-1")
        limiter.check("user-1")

        with pytest.raises(HTTPException):
            limiter.check("user-1")

        # Fast-forward past the 60s window
        mock_time.monotonic.return_value = 1061.0
        mock_time.time.return_value = 1700000061.0
        info = limiter.check("user-1")
        assert info.remaining == 1


def test_rate_limiter_cleanup() -> None:
    """Test that stale user entries are cleaned up periodically."""
    limiter = RateLimiter(rpm=100)
    limiter._CLEANUP_INTERVAL = 3  # trigger cleanup quickly

    with patch("any_llm.gateway.rate_limit.time") as mock_time:
        mock_time.monotonic.return_value = 1000.0
        mock_time.time.return_value = 1700000000.0

        # Create entries for stale-user
        limiter.check("stale-user")
        limiter.check("stale-user")

        # Fast-forward so stale-user's timestamps expire
        mock_time.monotonic.return_value = 1061.0
        mock_time.time.return_value = 1700000061.0

        # Third call triggers cleanup (interval=3)
        limiter.check("active-user")

        assert "stale-user" not in limiter._requests
        assert "active-user" in limiter._requests


def test_config_rejects_zero_rate_limit() -> None:
    """Test that rate_limit_rpm=0 is rejected by validation."""
    with pytest.raises(ValidationError):
        GatewayConfig(rate_limit_rpm=0)


def test_config_rejects_negative_rate_limit() -> None:
    """Test that rate_limit_rpm=-1 is rejected by validation."""
    with pytest.raises(ValidationError):
        GatewayConfig(rate_limit_rpm=-1)


def test_config_accepts_positive_rate_limit() -> None:
    """Test that rate_limit_rpm=1 is accepted."""
    config = GatewayConfig(rate_limit_rpm=1)
    assert config.rate_limit_rpm == 1


def test_config_accepts_none_rate_limit() -> None:
    """Test that rate_limit_rpm=None (disabled) is accepted."""
    config = GatewayConfig(rate_limit_rpm=None)
    assert config.rate_limit_rpm is None


class _MockCompletionError(Exception):
    """Raised to short-circuit acompletion in integration tests."""


def _make_rate_limit_client(
    postgres_url: str,
    rate_limit_rpm: int | None,
) -> Generator[TestClient]:
    """Create a TestClient with the given rate_limit_rpm config."""
    config = GatewayConfig(
        database_url=postgres_url,
        master_key="test-master-key",
        host="127.0.0.1",
        port=8000,
        auto_migrate=False,
        rate_limit_rpm=rate_limit_rpm,
    )
    _run_alembic_migrations(postgres_url)
    engine = create_engine(postgres_url, pool_pre_ping=True)
    app = create_app(config)

    def override_get_db() -> Generator[Session]:
        testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        Base.metadata.drop_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS alembic_version CASCADE"))
            conn.commit()


@pytest.fixture
def rate_limit_client(postgres_url: str) -> Generator[TestClient]:
    """TestClient with rate_limit_rpm=3."""
    yield from _make_rate_limit_client(postgres_url, rate_limit_rpm=3)


@pytest.fixture
def no_rate_limit_client(postgres_url: str) -> Generator[TestClient]:
    """TestClient with rate limiting disabled."""
    yield from _make_rate_limit_client(postgres_url, rate_limit_rpm=None)


def _create_test_user(client: TestClient, master_key: str = "test-master-key") -> str:
    """Create a test user and return the user_id."""
    header = {API_KEY_HEADER: f"Bearer {master_key}"}
    resp = client.post("/v1/users", json={"user_id": "rl-test-user", "alias": "RL"}, headers=header)
    assert resp.status_code == 200
    result: str = resp.json()["user_id"]
    return result


def _chat_request(client: TestClient, user_id: str, master_key: str = "test-master-key") -> Any:
    """Send a chat completion request using the master key."""
    header = {API_KEY_HEADER: f"Bearer {master_key}"}
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "openai:gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "user": user_id,
        },
        headers=header,
    )


def test_rate_limit_headers_on_success(rate_limit_client: TestClient) -> None:
    """Test that successful responses include rate limit headers."""
    from any_llm.types.completion import ChatCompletion, ChatCompletionMessage, Choice, CompletionUsage

    user_id = _create_test_user(rate_limit_client)

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
        usage=CompletionUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6),
    )

    async def mock_acompletion(**kwargs: Any) -> ChatCompletion:
        return mock_response

    with patch("any_llm.gateway.routes.chat.acompletion", new=mock_acompletion):
        resp = _chat_request(rate_limit_client, user_id)

    assert resp.status_code == 200
    assert resp.headers["X-RateLimit-Limit"] == "3"
    assert resp.headers["X-RateLimit-Remaining"] == "2"
    assert "X-RateLimit-Reset" in resp.headers
    # Reset should be a plausible Unix timestamp (wall-clock based)
    reset = int(resp.headers["X-RateLimit-Reset"])
    assert reset > 1_000_000_000


def test_rate_limit_returns_429_with_retry_after(rate_limit_client: TestClient) -> None:
    """Test that exceeding rate limit returns 429 with Retry-After."""
    user_id = _create_test_user(rate_limit_client)

    async def mock_acompletion(**kwargs: Any) -> None:
        raise _MockCompletionError

    with patch("any_llm.gateway.routes.chat.acompletion", new=mock_acompletion):
        # Exhaust the limit (rpm=3)
        for _ in range(3):
            _chat_request(rate_limit_client, user_id)

        # Next request should be rate-limited
        resp = _chat_request(rate_limit_client, user_id)

    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_no_rate_limit_allows_unlimited(no_rate_limit_client: TestClient) -> None:
    """Test that requests are not limited when rate_limit_rpm is None."""
    user_id = _create_test_user(no_rate_limit_client)

    async def mock_acompletion(**kwargs: Any) -> None:
        raise _MockCompletionError

    with patch("any_llm.gateway.routes.chat.acompletion", new=mock_acompletion):
        # Should never get 429
        for _ in range(10):
            resp = _chat_request(no_rate_limit_client, user_id)
            assert resp.status_code != 429
