# ruff: noqa: E402
from __future__ import annotations

import httpx
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
    original = ClientError(code=401, response_json={"error": {"message": "Invalid API key provided"}})

    result = convert_exception(original, "gemini")

    assert isinstance(result, AuthenticationError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original


def test_client_error_with_invalid_request() -> None:
    original = ClientError(code=400, response_json={"error": {"message": "Invalid request: bad parameter"}})

    result = convert_exception(original, "gemini")

    assert isinstance(result, InvalidRequestError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original


def test_server_error_conversion() -> None:
    original = ServerError(code=500, response_json={"error": {"message": "Internal server error"}})

    result = convert_exception(original, "gemini")

    assert isinstance(result, ProviderError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original


def test_api_error_with_rate_limit_message() -> None:
    original = APIError(code=429, response_json={"error": {"message": "Rate limit exceeded. Too many requests."}})

    result = convert_exception(original, "gemini")

    assert isinstance(result, RateLimitError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original


def test_api_error_generic() -> None:
    original = APIError(code=400, response_json={"error": {"message": "Some API error occurred"}})

    result = convert_exception(original, "gemini")

    assert isinstance(result, ProviderError)
    assert result.provider_name == "gemini"
    assert result.original_exception is original


def test_client_error_when_model_rejects_zero_thinking_budget() -> None:
    """Models that only work in thinking mode reject thinking_budget=0 with a 400."""
    original = ClientError(
        code=400,
        response_json={"error": {"message": "Budget 0 is invalid. This model only works in thinking mode."}},
    )

    result = convert_exception(original, "gemini")

    assert isinstance(result, InvalidRequestError)
    assert "Budget 0 is invalid" in str(result)
    assert result.original_exception is original


def test_client_error_when_model_rejects_minimal_thinking_level() -> None:
    """Models without a MINIMAL thinking level reject the clamp with a 400."""
    original = ClientError(
        code=400,
        response_json={
            "error": {
                "message": "Thinking level MINIMAL is not supported for this model. Please retry with other thinking level."
            }
        },
        response=httpx.Response(400),
    )

    result = convert_exception(original, "gemini")

    assert isinstance(result, InvalidRequestError)
    assert "Thinking level MINIMAL is not supported" in str(result)
    assert result.original_exception is original


def test_client_error_without_attached_response_still_maps_to_invalid_request() -> None:
    """With no usable response object, the status Google puts in the error body still classifies the 400."""
    original = ClientError(
        code=400,
        response_json={
            "error": {
                "code": 400,
                "message": "Thinking level MINIMAL is not supported for this model. Please retry with other thinking level.",
                "status": "INVALID_ARGUMENT",
            }
        },
    )

    result = convert_exception(original, "gemini")

    assert isinstance(result, InvalidRequestError)
    assert "Thinking level MINIMAL is not supported" in str(result)
