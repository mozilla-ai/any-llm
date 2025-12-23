from __future__ import annotations

import functools
import os
import re
import warnings
from typing import TYPE_CHECKING, Any, TypeVar

from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError,
    ContentFilterError,
    ContextLengthExceededError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)

if TYPE_CHECKING:
    from collections.abc import Callable

F = TypeVar("F", bound="Callable[..., Any]")


ANY_LLM_UNIFIED_EXCEPTIONS_ENV = "ANY_LLM_UNIFIED_EXCEPTIONS"

_DEPRECATION_WARNING = (
    "Provider-specific exceptions will be converted to unified any-llm exceptions "
    "(e.g., RateLimitError, AuthenticationError) in a future version. "
    "To enable this behavior now, set the environment variable ANY_LLM_UNIFIED_EXCEPTIONS=1. "
    "The original exception will be available in the original_exception attribute."
)


def convert_exception(
    exception: Exception,
    provider_name: str,
) -> AnyLLMError:
    """Convert a provider-specific exception to an AnyLLMError.

    This function attempts to classify the exception based on its type name
    and message content, converting it to the appropriate AnyLLMError subclass.

    Args:
        exception: The original exception from the SDK
        provider_name: Name of the provider that raised the exception

    Returns:
        An AnyLLMError subclass instance

    """
    if isinstance(exception, AnyLLMError):
        return exception

    original_message = str(exception)
    exc_text = f"{type(exception).__name__.lower()} {str(exception).lower()}"

    if re.search(r"ratelimit|rate_limit|too many requests|rate limit|quota exceeded", exc_text):
        return RateLimitError(
            message=original_message,
            original_exception=exception,
            provider_name=provider_name,
        )

    if re.search(
        r"auth|permission|invalid api key|invalid key|unauthorized|authentication|"
        r"permission denied|access denied|forbidden|invalid_api_key|api key not found|api key invalid|"
        r"api key not valid|incorrect api key|not valid.*api key",
        exc_text,
    ):
        return AuthenticationError(
            message=original_message,
            original_exception=exception,
            provider_name=provider_name,
        )

    if re.search(r"context.*length|length.*context|token limit|maximum.*length", exc_text):
        return ContextLengthExceededError(
            message=original_message,
            original_exception=exception,
            provider_name=provider_name,
        )

    if re.search(r"notfound|not_found|model not found|does not exist|model.*not.*found", exc_text):
        return ModelNotFoundError(
            message=original_message,
            original_exception=exception,
            provider_name=provider_name,
        )

    if re.search(
        r"content.*(filter|policy)|(filter|policy).*content|safety|moderation|blocked|harmful content",
        exc_text,
    ):
        return ContentFilterError(
            message=original_message,
            original_exception=exception,
            provider_name=provider_name,
        )

    if re.search(r"invalid|badrequest|validation", exc_text):
        return InvalidRequestError(
            message=original_message,
            original_exception=exception,
            provider_name=provider_name,
        )

    if re.search(
        r"timeout|connection|network|server|internal|service|service unavailable",
        exc_text,
    ):
        return ProviderError(
            message=original_message,
            original_exception=exception,
            provider_name=provider_name,
        )

    return ProviderError(
        message=original_message,
        original_exception=exception,
        provider_name=provider_name,
    )


def _handle_exception(exception: Exception, provider_name: str) -> None:
    """Handle an exception based on the unified exceptions flag.

    Args:
        exception: The original exception
        provider_name: Name of the provider for error context

    Raises:
        AnyLLMError: If unified exceptions are enabled
        Exception: The original exception if unified exceptions are disabled

    """
    if os.environ.get(ANY_LLM_UNIFIED_EXCEPTIONS_ENV, "").lower() in ("1", "true", "yes", "on"):
        converted = convert_exception(exception, provider_name)
        raise converted from exception

    warnings.warn(
        _DEPRECATION_WARNING,
        DeprecationWarning,
        stacklevel=4,  # Point to the user's code, not internal handlers
    )
    raise exception


def handle_exceptions(*, wrap_streaming: bool = False) -> Callable[[F], F]:
    """Handle exceptions in async methods.

    This decorator wraps async methods to catch provider-specific exceptions
    and convert them to unified AnyLLMError subclasses (when enabled).
    It expects the decorated method to be a method on a class with a
    `PROVIDER_NAME` attribute.

    Args:
        wrap_streaming: If True, the result will be wrapped with an async iterator
            wrapper if it's an async iterator. This is useful for streaming responses
            where exceptions may occur during iteration.

    Returns:
        A decorator function.

    """

    def decorator(func: F) -> F:
        if wrap_streaming:

            async def _wrap_async_iterator(
                async_iter: Any,
                provider_name: str,
            ) -> Any:
                """Wrap an async iterator to handle exceptions during iteration."""
                try:
                    async for item in async_iter:
                        yield item
                except Exception as e:
                    _handle_exception(e, provider_name)

            @functools.wraps(func)
            async def streaming_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                provider_name = getattr(self, "PROVIDER_NAME", "unknown")
                try:
                    result = await func(self, *args, **kwargs)
                except Exception as e:
                    _handle_exception(e, provider_name)
                    return None  # unreachable, but helps type checkers

                # Check if result is an async iterator (streaming response)
                # If so, wrap it to handle exceptions during iteration
                if hasattr(result, "__aiter__"):
                    return _wrap_async_iterator(result, provider_name)

                # Non-streaming response, return as-is
                return result

            return streaming_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            provider_name = getattr(self, "PROVIDER_NAME", "unknown")
            try:
                return await func(self, *args, **kwargs)
            except Exception as e:
                _handle_exception(e, provider_name)

        return wrapper  # type: ignore[return-value]

    return decorator
