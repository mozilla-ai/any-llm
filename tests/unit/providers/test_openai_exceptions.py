# ruff: noqa: E402
from __future__ import annotations

from unittest.mock import MagicMock

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
    original.status_code = 500

    result = convert_exception(original, "openai")

    assert isinstance(result, ProviderError)
    assert result.provider_name == "openai"
    assert result.original_exception is original
