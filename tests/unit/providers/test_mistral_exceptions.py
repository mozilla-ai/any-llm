# ruff: noqa: E402
from __future__ import annotations

import pytest

mistralai = pytest.importorskip("mistralai")

from mistralai.models import HTTPValidationError, SDKError

from any_llm.exceptions import (
    InvalidRequestError,
    ProviderError,
)
from any_llm.utils.exception_handler import convert_exception


def test_sdk_error_with_validation_message() -> None:
    original = SDKError("Validation error: invalid parameter", status_code=400, body="Invalid parameter")

    result = convert_exception(original, "mistral")

    assert isinstance(result, InvalidRequestError)
    assert result.provider_name == "mistral"
    assert result.original_exception is original


def test_sdk_error_with_server_error() -> None:
    original = SDKError("Internal server error", status_code=500, body="Server error")

    result = convert_exception(original, "mistral")

    assert isinstance(result, ProviderError)
    assert result.provider_name == "mistral"
    assert result.original_exception is original


def test_http_validation_error_conversion() -> None:
    original = HTTPValidationError(detail=[])

    result = convert_exception(original, "mistral")

    assert isinstance(result, InvalidRequestError)
    assert result.provider_name == "mistral"
    assert result.original_exception is original
