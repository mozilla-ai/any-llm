# ruff: noqa: E402
from __future__ import annotations

from unittest.mock import MagicMock

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


def _quota_error(*details: object, headers: dict[str, str] | None = None) -> ClientError:
    """A 429 shaped like Gemini's, its retry delay in the body's RetryInfo rather than a header."""
    body = {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota, please check your plan and billing details.",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": []},
                *details,
            ],
        }
    }
    # google-genai's async path attaches the aiohttp response, which spells the status `status`
    return ClientError(code=429, response_json=body, response=MagicMock(status=429, headers=headers or {}))


@pytest.mark.parametrize(("delay", "expected"), [("20s", "20"), ("1.5s", "1.5"), ("0s", "0")])
def test_rate_limit_reads_retry_delay_from_retry_info(delay: str, expected: str) -> None:
    original = _quota_error({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay})

    result = convert_exception(original, "gemini")

    assert isinstance(result, RateLimitError)
    assert result.retry_after == expected


@pytest.mark.parametrize(
    "details",
    [
        (),
        ({"@type": "type.googleapis.com/google.rpc.RetryInfo"},),
        ({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "soon"},),
        ({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "-5s"},),
        ({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1e1000000s"},),
        ({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": 20},),
        ({"@type": "type.googleapis.com/google.rpc.Help", "retryDelay": "20s"},),
    ],
    ids=["no-retry-info", "no-delay", "not-a-duration", "negative", "overflowing", "not-a-string", "other-type"],
)
def test_rate_limit_without_a_usable_retry_delay_has_no_retry_after(details: tuple[object, ...]) -> None:
    result = convert_exception(_quota_error(*details), "gemini")

    assert isinstance(result, RateLimitError)
    assert result.retry_after is None


@pytest.mark.parametrize(("headers", "expected"), [({"retry-after": "7"}, "7"), ({"retry-after-ms": "1500"}, "1.5")])
def test_a_retry_header_wins_over_retry_info(headers: dict[str, str], expected: str) -> None:
    original = _quota_error({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "20s"}, headers=headers)

    result = convert_exception(original, "gemini")

    assert isinstance(result, RateLimitError)
    assert result.retry_after == expected
