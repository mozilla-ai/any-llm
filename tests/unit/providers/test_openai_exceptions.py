# ruff: noqa: E402
from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

openai = pytest.importorskip("openai")

from openai import APIError as OpenAIAPIError
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import BadRequestError as OpenAIBadRequestError
from openai import NotFoundError as OpenAINotFoundError
from openai import RateLimitError as OpenAIRateLimitError

from any_llm.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from any_llm.utils.exception_handler import convert_exception


def test_rate_limit_error_conversion() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}

    original = OpenAIRateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )

    result = convert_exception(original, "openai")

    assert isinstance(result, RateLimitError)
    assert result.provider_name == "openai"
    assert result.original_exception is original


def test_auth_error_conversion() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}

    original = OpenAIAuthenticationError(
        message="Invalid API key",
        response=mock_response,
        body={"error": {"message": "Invalid API key"}},
    )

    result = convert_exception(original, "openai")

    assert isinstance(result, AuthenticationError)
    assert result.provider_name == "openai"
    assert result.original_exception is original


def test_bad_request_error_conversion() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.headers = {}

    original = OpenAIBadRequestError(
        message="Invalid parameter",
        response=mock_response,
        body={"error": {"message": "Invalid parameter"}},
    )

    result = convert_exception(original, "openai")

    assert isinstance(result, InvalidRequestError)
    assert result.provider_name == "openai"
    assert result.original_exception is original


def test_not_found_error_conversion() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.headers = {}

    original = OpenAINotFoundError(
        message="Model not found",
        response=mock_response,
        body={"error": {"message": "Model not found"}},
    )

    result = convert_exception(original, "openai")

    assert isinstance(result, ModelNotFoundError)
    assert result.provider_name == "openai"
    assert result.original_exception is original


def test_api_error_with_500_status() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.headers = {}

    original = OpenAIAPIError(
        message="Internal server error",
        request=MagicMock(),
        body={"error": {"message": "Internal server error"}},
    )

    result = convert_exception(original, "openai")

    assert isinstance(result, ProviderError)
    assert result.provider_name == "openai"
    assert result.original_exception is original


def test_status_code_and_param_are_preserved_on_the_unified_error() -> None:
    """A consumer can classify a rejection without unwrapping original_exception.

    The OpenAI client unwraps the ``{"error": {...}}`` response body before
    constructing the exception, so the SDK populates ``param``/``code``/``type``
    as attributes. This is the reasoning_effort case from mozilla-ai/otari#331:
    a gateway needs status_code plus param to surface the actionable remedy.
    """
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.headers = {}

    original = OpenAIBadRequestError(
        message=(
            "Function tools with reasoning_effort are not supported for gpt-5.6-sol in "
            "/v1/chat/completions. To use function tools, use /v1/responses or set "
            "reasoning_effort to 'none'."
        ),
        response=mock_response,
        body={
            "message": "Function tools with reasoning_effort are not supported",
            "type": "invalid_request_error",
            "param": "reasoning_effort",
            "code": None,
        },
    )

    result = convert_exception(original, "openai")

    assert isinstance(result, InvalidRequestError)
    assert result.status_code == 400
    assert result.param == "reasoning_effort"
    assert result.error_type == "invalid_request_error"
    assert result.code is None


def test_rate_limit_metadata_is_preserved_on_the_unified_error() -> None:
    """RateLimitError overrides __init__, so it has to forward the new fields."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}

    original = OpenAIRateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"message": "Rate limit exceeded", "type": "rate_limit_error", "code": "rate_limit_exceeded"},
    )

    result = convert_exception(original, "openai")

    assert isinstance(result, RateLimitError)
    assert result.status_code == 429
    assert result.code == "rate_limit_exceeded"
    assert result.error_type == "rate_limit_error"


def test_retry_after_is_read_from_a_real_httpx_response() -> None:
    """Pins the case-insensitive httpx.Headers lookup, not just a dict.

    The SDK attaches a real httpx.Response, so the header arrives however the
    provider capitalized it on the wire.
    """
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        headers={"Retry-After": "30"},
    )

    original = OpenAIRateLimitError(
        message="Rate limit exceeded",
        response=response,
        body={"message": "Rate limit exceeded", "type": "rate_limit_error"},
    )

    result = convert_exception(original, "openai")

    assert isinstance(result, RateLimitError)
    assert result.status_code == 429
    assert result.retry_after == "30"
