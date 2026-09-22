# ruff: noqa: E402
from __future__ import annotations

import pytest

google_genai = pytest.importorskip("google.genai")

from google.genai.errors import APIError, ClientError, ServerError

from any_llm.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from any_llm.utils.exception_handler import convert_exception


def test_client_error_with_invalid_api_key() -> None:
    original = ClientError(code=401, response_json={"error": {"message": "Invalid API key provided"}})

    result = convert_exception(original, "gemini")

    assert isinstance(result, AuthenticationError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original
    assert result.status_code == 401


def test_client_error_with_invalid_request() -> None:
    original = ClientError(code=400, response_json={"error": {"message": "Invalid request: bad parameter"}})

    result = convert_exception(original, "gemini")

    assert isinstance(result, InvalidRequestError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original
    assert result.status_code == 400


def test_server_error_conversion() -> None:
    original = ServerError(code=500, response_json={"error": {"message": "Internal server error"}})

    result = convert_exception(original, "gemini")

    assert isinstance(result, ProviderError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original
    assert result.status_code == 500


def test_api_error_with_rate_limit_message() -> None:
    original = APIError(code=429, response_json={"error": {"message": "Rate limit exceeded. Too many requests."}})

    result = convert_exception(original, "gemini")

    assert isinstance(result, RateLimitError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original
    assert result.status_code == 429


def test_api_error_generic() -> None:
    original = APIError(code=400, response_json={"error": {"message": "Some API error occurred"}})

    result = convert_exception(original, "gemini")

    assert isinstance(result, InvalidRequestError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original
    assert result.status_code == 400


def test_client_error_404_is_model_not_found_without_file_context() -> None:
    original = ClientError(code=404, response_json={"error": {"message": "File not found", "status": "NOT_FOUND"}})

    result = convert_exception(original, "gemini")

    assert isinstance(result, ModelNotFoundError)
    assert result.status_code == 404
