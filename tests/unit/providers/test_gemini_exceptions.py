# ruff: noqa: E402
from __future__ import annotations

import pytest

google_genai = pytest.importorskip("google.genai")

from google.genai.errors import APIError, ClientError, ServerError

from any_llm.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
)
from any_llm.utils.exception_handler import convert_exception


def test_client_error_with_invalid_api_key() -> None:
    original = ClientError("Invalid API key provided")

    result = convert_exception(original, "gemini")

    assert isinstance(result, AuthenticationError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original


def test_client_error_with_invalid_request() -> None:
    original = ClientError("Invalid request: bad parameter")

    result = convert_exception(original, "gemini")

    assert isinstance(result, InvalidRequestError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original


def test_server_error_conversion() -> None:
    original = ServerError("Internal server error")

    result = convert_exception(original, "gemini")

    assert isinstance(result, ProviderError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original


def test_api_error_with_rate_limit_message() -> None:
    original = APIError("Rate limit exceeded. Too many requests.")

    result = convert_exception(original, "gemini")

    assert isinstance(result, RateLimitError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original


def test_api_error_generic() -> None:
    original = APIError("Some API error occurred")

    result = convert_exception(original, "gemini")

    assert isinstance(result, ProviderError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original
