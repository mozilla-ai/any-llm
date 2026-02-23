"""Tests for per-user rate limiting."""

import pytest

from any_llm.gateway.rate_limit import RateLimiter


def test_rate_limiter_allows_under_limit() -> None:
    """Test that requests under the limit are allowed."""
    limiter = RateLimiter(rpm=5)
    for _ in range(5):
        limiter.check("user-1")


def test_rate_limiter_rejects_over_limit() -> None:
    """Test that requests over the limit are rejected with 429."""
    from fastapi import HTTPException

    limiter = RateLimiter(rpm=3)
    for _ in range(3):
        limiter.check("user-1")

    with pytest.raises(HTTPException) as exc_info:
        limiter.check("user-1")
    assert exc_info.value.status_code == 429


def test_rate_limiter_per_user_isolation() -> None:
    """Test that rate limits are tracked independently per user."""
    limiter = RateLimiter(rpm=2)
    limiter.check("user-1")
    limiter.check("user-1")

    # user-2 should still be allowed
    limiter.check("user-2")
    limiter.check("user-2")

    # user-1 should be blocked, user-2 also blocked
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        limiter.check("user-1")
    with pytest.raises(HTTPException):
        limiter.check("user-2")
