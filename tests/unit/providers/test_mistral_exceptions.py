# ruff: noqa: E402
from __future__ import annotations

import httpx
import pytest

mistralai = pytest.importorskip("mistralai")

from mistralai.models import HTTPValidationError, SDKError

from any_llm.exceptions import (
    InvalidRequestError,
    ProviderError,
)
from any_llm.utils.exception_handler import convert_exception


def test_sdk_error_with_validation_message() -> None:
    mock_response = httpx.Response(status_code=400, content=b"Invalid parameter")
    original = SDKError("Validation error: invalid parameter", mock_response, body="Invalid parameter")

    result = convert_exception(original, "mistral")

    assert isinstance(result, InvalidRequestError)
    assert result.provider_name == "mistral"
    assert result.original_exception is original


def test_sdk_error_with_server_error() -> None:
    mock_response = httpx.Response(status_code=500, content=b"Server error")
    original = SDKError("Internal server error", mock_response, body="Server error")

    result = convert_exception(original, "mistral")

    assert isinstance(result, ProviderError)
    assert result.provider_name == "mistral"
    assert result.original_exception is original


def test_http_validation_error_conversion() -> None:
    from mistralai.models.httpvalidationerror import HTTPValidationErrorData

    mock_response = httpx.Response(status_code=422, content=b"Validation error")
    data = HTTPValidationErrorData()
    original = HTTPValidationError(data=data, raw_response=mock_response, body="Validation error")

    result = convert_exception(original, "mistral")

    assert isinstance(result, InvalidRequestError)
    assert result.provider_name == "mistral"
    assert result.original_exception is original
