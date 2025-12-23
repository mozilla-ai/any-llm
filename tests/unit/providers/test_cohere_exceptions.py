# ruff: noqa: E402
from __future__ import annotations

import pytest

cohere = pytest.importorskip("cohere")

from cohere import (
    BadRequestError,
    ForbiddenError,
    GatewayTimeoutError,
    InternalServerError,
    NotFoundError,
    ServiceUnavailableError,
    TooManyRequestsError,
    UnauthorizedError,
)

from any_llm.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from any_llm.utils.exception_handler import convert_exception


def test_too_many_requests_error_conversion() -> None:
    original = TooManyRequestsError(body="Rate limit exceeded")

    result = convert_exception(original, "cohere")

    assert isinstance(result, RateLimitError)
    assert result.provider_name == "cohere"
    assert result.original_exception is original


def test_unauthorized_error_conversion() -> None:
    original = UnauthorizedError(body="Unauthorized")

    result = convert_exception(original, "cohere")

    assert isinstance(result, AuthenticationError)
    assert result.provider_name == "cohere"
    assert result.original_exception is original


def test_forbidden_error_conversion() -> None:
    original = ForbiddenError(body="Forbidden")

    result = convert_exception(original, "cohere")

    assert isinstance(result, AuthenticationError)
    assert result.provider_name == "cohere"
    assert result.original_exception is original


def test_bad_request_error_conversion() -> None:
    original = BadRequestError(body="Bad request")

    result = convert_exception(original, "cohere")

    assert isinstance(result, InvalidRequestError)
    assert result.provider_name == "cohere"
    assert result.original_exception is original


def test_not_found_error_conversion() -> None:
    original = NotFoundError(body="Not found")

    result = convert_exception(original, "cohere")

    assert isinstance(result, ModelNotFoundError)
    assert result.provider_name == "cohere"
    assert result.original_exception is original


def test_internal_server_error_conversion() -> None:
    original = InternalServerError(body="Internal server error")

    result = convert_exception(original, "cohere")

    assert isinstance(result, ProviderError)
    assert result.provider_name == "cohere"
    assert result.original_exception is original


def test_service_unavailable_error_conversion() -> None:
    original = ServiceUnavailableError(body="Service unavailable")

    result = convert_exception(original, "cohere")

    assert isinstance(result, ProviderError)
    assert result.provider_name == "cohere"
    assert result.original_exception is original


def test_gateway_timeout_error_conversion() -> None:
    original = GatewayTimeoutError(body="Gateway timeout")

    result = convert_exception(original, "cohere")

    assert isinstance(result, ProviderError)
    assert result.provider_name == "cohere"
    assert result.original_exception is original
