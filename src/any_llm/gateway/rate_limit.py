"""In-memory per-user rate limiter using a sliding window."""

import time
from collections import defaultdict

from fastapi import HTTPException, status


class RateLimiter:
    """Simple sliding-window rate limiter.

    Tracks request timestamps per user and rejects requests that exceed
    the configured requests-per-minute (RPM) limit.
    """

    def __init__(self, rpm: int) -> None:
        self._rpm = rpm
        self._window_sec = 60.0
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, user_id: str) -> None:
        """Check whether a request is allowed for the given user.

        Raises:
            HTTPException: 429 if the rate limit has been exceeded

        """
        now = time.monotonic()
        cutoff = now - self._window_sec

        timestamps = self._requests[user_id]
        # Prune expired entries
        self._requests[user_id] = [t for t in timestamps if t > cutoff]

        if len(self._requests[user_id]) >= self._rpm:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

        self._requests[user_id].append(now)
