from __future__ import annotations

import functools
import os
import re
import warnings
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

from pydantic import ValidationError

from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError,
    ContentFilterError,
    ContextLengthExceededError,
    GatewayTimeoutError,
    InsufficientFundsError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    UpstreamProviderError,
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


# Message/type-name patterns tried in order; the first match wins, so more
# specific patterns must stay ahead of broader ones (for example the API-key
# patterns before the bare "invalid"). Anything unmatched becomes a ProviderError.
_ERROR_PATTERNS: tuple[tuple[str, type[AnyLLMError]], ...] = (
    (r"ratelimit|rate_limit|too many requests|rate limit|quota exceeded", RateLimitError),
    (
        r"auth|permission|invalid api key|invalid key|unauthorized|authentication|"
        r"permission denied|access denied|forbidden|invalid_api_key|api key not found|api key invalid|"
        r"api key not valid|incorrect api key|not valid.*api key|"
        r"api key.*invalid|invalid.*api key",
        AuthenticationError,
    ),
    (r"context.*length|length.*context|token limit|maximum.*length", ContextLengthExceededError),
    (r"notfound|not_found|model not found|does not exist|model.*not.*found", ModelNotFoundError),
    (
        r"content.*(filter|policy)|(filter|policy).*content|safety|moderation|blocked|harmful content",
        ContentFilterError,
    ),
    (r"invalid|badrequest|validation", InvalidRequestError),
    (r"insufficient.*funds|payment.*required|budget.*exceeded", InsufficientFundsError),
    (r"bad.*gateway|upstream.*error|upstream.*provider", UpstreamProviderError),
    (r"gateway.*timeout", GatewayTimeoutError),
    (r"timeout|connection|network|server|internal|service|service unavailable", ProviderError),
)

# Statuses that identify the failure on their own. Checked before the message
# patterns, because the status is what the provider actually returned while a
# message match is a guess: a 500 whose body happens to mention "policy" is a
# provider outage, not a content-filter rejection. The 402/502/504 entries also
# align the code with what those exception docstrings already claim.
_STATUS_ERROR_CLASSES: dict[int, type[AnyLLMError]] = {
    401: AuthenticationError,
    402: InsufficientFundsError,
    403: AuthenticationError,
    404: ModelNotFoundError,
    429: RateLimitError,
    502: UpstreamProviderError,
    504: GatewayTimeoutError,
}

# "The request was rejected", where the status says that much but only the
# message says why. ContextLengthExceededError and ContentFilterError have no
# status of their own (OpenAI returns 400 for both), and some providers report a
# bad API key as a 400, so the patterns still decide here. InvalidRequestError
# is the floor when nothing more specific matches.
_REJECTED_REQUEST_STATUSES = frozenset({400, 422})


def _classify_by_message(exc_text: str) -> type[AnyLLMError]:
    """First matching pattern in the ordered table, else ProviderError."""
    for pattern, error_class in _ERROR_PATTERNS:
        if re.search(pattern, exc_text):
            return error_class
    return ProviderError


def _classify(exc_text: str, status_code: int | None) -> type[AnyLLMError]:
    """Pick the unified exception class for a failure.

    Prefers the HTTP status where it is unambiguous and falls back to the
    message patterns otherwise, so a provider that returns an unhelpful body
    still classifies correctly and a message coincidence cannot override a
    status the provider actually sent.
    """
    if status_code is None:
        return _classify_by_message(exc_text)

    keyed = _STATUS_ERROR_CLASSES.get(status_code)
    if keyed is not None:
        return keyed

    if status_code in _REJECTED_REQUEST_STATUSES:
        by_message = _classify_by_message(exc_text)
        # ProviderError is the table's "nothing matched" outcome. A rejected
        # request is the caller's problem, not a provider fault.
        return InvalidRequestError if by_message is ProviderError else by_message

    if status_code >= 500:
        return ProviderError

    # Any other status (3xx, 409, 413, ...) carries no agreed meaning here, so
    # the message patterns stay in charge.
    return _classify_by_message(exc_text)


class _ErrorMetadata(NamedTuple):
    """Structured HTTP fields recovered from a provider SDK exception."""

    status_code: int | None
    code: str | None
    param: str | None
    error_type: str | None


def _extract_status_code(exception: Exception) -> int | None:
    """Read the HTTP status off the exception, or off its attached response."""
    status_code = getattr(exception, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exception, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status

    return None


def _error_body_dicts(exception: Exception) -> tuple[dict[str, Any], ...]:
    """Dicts to read structured error fields from, in precedence order.

    The OpenAI client unwraps a ``{"error": {...}}`` response body before
    constructing the exception, so the fields usually sit at the top level of
    ``body``. Other SDKs keep the nesting, and there the inner ``error`` dict is
    the authoritative detail while the outer dict is only an envelope: Anthropic
    sends a literal ``{"type": "error", "error": {"type": "invalid_request_error"}}``,
    so the nested dict has to win to avoid reporting ``error_type="error"``.
    """
    body = getattr(exception, "body", None)
    if not isinstance(body, dict):
        return ()

    nested = body.get("error")
    if isinstance(nested, dict):
        return (nested, body)
    return (body,)


def _extract_string_field(exception: Exception, name: str, bodies: tuple[dict[str, Any], ...]) -> str | None:
    """Read a string field off the exception, falling back to the response body."""
    value = getattr(exception, name, None)
    if isinstance(value, str):
        return value

    for body in bodies:
        body_value = body.get(name)
        if isinstance(body_value, str):
            return body_value

    return None


def _extract_error_metadata(exception: Exception) -> _ErrorMetadata:
    """Pull structured HTTP metadata off a provider SDK exception, best-effort.

    Provider-agnostic and non-raising: every field is optional, and a provider
    that does not report one (or a non-HTTP failure such as a timeout) simply
    yields ``None`` for it. ``getattr`` is used deliberately here because the
    incoming exception is an arbitrary third-party SDK type whose attributes are
    not statically known.
    """
    bodies = _error_body_dicts(exception)
    return _ErrorMetadata(
        status_code=_extract_status_code(exception),
        code=_extract_string_field(exception, "code", bodies),
        param=_extract_string_field(exception, "param", bodies),
        error_type=_extract_string_field(exception, "type", bodies),
    )


def _extract_retry_after(exception: Exception) -> str | None:
    """Read the ``Retry-After`` header off the exception's attached response.

    Returned verbatim, because the header is either a number of seconds or an
    HTTP-date and normalizing it would throw away the caller's ability to tell
    the two apart. ``httpx.Headers`` lookups are case-insensitive, but a plain
    dict is not, so both spellings are tried.
    """
    headers = getattr(getattr(exception, "response", None), "headers", None)
    get_header = getattr(headers, "get", None)
    if not callable(get_header):
        return None

    for name in ("retry-after", "Retry-After"):
        value = get_header(name)
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return str(value)

    return None


def convert_exception(
    exception: Exception,
    provider_name: str,
) -> AnyLLMError:
    """Convert a provider-specific exception to an AnyLLMError.

    Classifies by the HTTP status the provider returned where that is
    unambiguous, falling back to the exception's type name and message content
    otherwise. Structured HTTP metadata the SDK exposed (``status_code``,
    ``code``, ``param``, ``type``) is carried onto the unified exception so
    consumers can classify a failure without unwrapping ``original_exception``,
    and a :class:`~any_llm.exceptions.RateLimitError` additionally carries the
    ``Retry-After`` header when the provider sent one.

    Args:
        exception: The original exception from the SDK
        provider_name: Name of the provider that raised the exception

    Returns:
        An AnyLLMError subclass instance

    """
    if isinstance(exception, AnyLLMError):
        return exception

    exc_text = f"{type(exception).__name__.lower()} {str(exception).lower()}"
    metadata = _extract_error_metadata(exception)
    error_class = _classify(exc_text, metadata.status_code)

    error = error_class(
        message=str(exception),
        original_exception=exception,
        provider_name=provider_name,
        status_code=metadata.status_code,
        code=metadata.code,
        param=metadata.param,
        error_type=metadata.error_type,
    )
    # Assigned after construction rather than passed in: retry_after exists only
    # on RateLimitError, and the shared construction site above is typed against
    # the base class. isinstance keeps the attribute access type-checked.
    if isinstance(error, RateLimitError):
        error.retry_after = _extract_retry_after(exception)
    return error


def _handle_exception(exception: Exception, provider_name: str) -> None:
    """Handle an exception based on the unified exceptions flag.

    Args:
        exception: The original exception
        provider_name: Name of the provider for error context

    Raises:
        AnyLLMError: If unified exceptions are enabled
        Exception: The original exception if unified exceptions are disabled
        pydantic.ValidationError: Always re-raised unchanged

    """
    # AnyLLMError subclasses are already unified exceptions and should not be re-processed
    # or trigger the deprecation warning.
    if isinstance(exception, AnyLLMError):
        raise exception

    # When using response_format with a Pydantic model, acompletion() deserializes the
    # response content via model_validate_json. If the LLM produces output that doesn't
    # conform to the schema, pydantic raises ValidationError. This should bubble up unchanged.
    # See: https://github.com/mozilla-ai/any-llm/issues/799
    if isinstance(exception, ValidationError):
        raise exception

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
