"""In-memory per-user rate limiter using a sliding window."""

import math
import time
from collections import defaultdict
from dataclasses import dataclass

from fastapi import HTTPException, Request, status


@dataclass
class RateLimitInfo:
    """Rate limit status returned by a successful check."""

    limit: int
    remaining: int
    reset: float


class RateLimiter:
    """Simple sliding-window rate limiter.

    Tracks request timestamps per user and rejects requests that exceed
    the configured requests-per-minute (RPM) limit.
    """

    _CLEANUP_INTERVAL = 1000

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._window_sec = 60.0
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._call_count = 0

    def check(self, user_id: str) -> RateLimitInfo:
        """Check whether a request is allowed for the given user.

        Returns:
            RateLimitInfo with limit, remaining, and reset timestamp

        Raises:
            HTTPException: 429 if the rate limit has been exceeded

        """
        now = time.monotonic()
        cutoff = now - self._window_sec

        timestamps = self._requests[user_id]
        self._requests[user_id] = [t for t in timestamps if t > cutoff]

        self._call_count += 1
        if self._call_count >= self._CLEANUP_INTERVAL:
            self._cleanup(cutoff)
            self._call_count = 0

        current = self._requests[user_id]
        if len(current) >= self._rpm:
            oldest = current[0]
            retry_after = math.ceil(oldest - cutoff)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(retry_after)},
            )

        current.append(now)
        remaining = self._rpm - len(current)
        # Use wall-clock time for the externally-facing reset header
        reset = time.time() + (current[0] + self._window_sec - now)
        return RateLimitInfo(limit=self._rpm, remaining=remaining, reset=reset)

    def _cleanup(self, cutoff: float) -> None:
        """Remove entries for users with no recent requests."""
        stale = [uid for uid, ts in self._requests.items() if all(t <= cutoff for t in ts)]
        for uid in stale:
            del self._requests[uid]


def check_rate_limit(request: Request, user_id: str) -> RateLimitInfo | None:
    """Check rate limit for a user, returning info for header injection.

    Returns RateLimitInfo when rate limiting is active, None when disabled.
    """
    rate_limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if rate_limiter is None:
        return None
    return rate_limiter.check(user_id)
