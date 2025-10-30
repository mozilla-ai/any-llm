"""Decorators for any-llm API functions."""

import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

BATCH_API_EXPERIMENTAL_MESSAGE = "The Batch API is experimental and may have breaking changes in future versions."
DEFAULT_EXPERIMENTAL_MESSAGE = "This API is experimental and subject to breaking changes."


def experimental(message: str = DEFAULT_EXPERIMENTAL_MESSAGE) -> Callable[[F], F]:
    """Mark a function as experimental.

    This decorator emits a FutureWarning when the decorated function is called,
    alerting users that the API may change in future versions.

    Args:
        message: Custom warning message to display

    Returns:
        Decorator function

    """

    def decorator(func: F) -> F:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, FutureWarning, stacklevel=2)
            return await func(*args, **kwargs)

        # Return appropriate wrapper based on whether function is async
        import inspect

        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator
