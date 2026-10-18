# ruff: noqa: E402
from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

mistralai = pytest.importorskip("mistralai")
# mistralai 3 builds its errors on httpx2 responses.
httpx2 = pytest.importorskip("httpx2")

from mistralai.client.errors import HTTPValidationError, HTTPValidationErrorData, ResponseValidationError, SDKError

from any_llm.exceptions import (
    InvalidRequestError,
    ProviderError,
)
from any_llm.utils.exception_handler import convert_exception


def test_sdk_error_with_validation_message() -> None:
    mock_response = httpx2.Response(status_code=400, content=b"Invalid parameter")
    original = SDKError("Validation error: invalid parameter", mock_response, body="Invalid parameter")

    result = convert_exception(original, "mistral")

    assert isinstance(result, InvalidRequestError)
    assert result.provider_name == "mistral"
    assert result.original_exception is original


def test_sdk_error_with_server_error() -> None:
    mock_response = httpx2.Response(status_code=500, content=b"Server error")
    original = SDKError("Internal server error", mock_response, body="Server error")

    result = convert_exception(original, "mistral")

    assert isinstance(result, ProviderError)
    assert result.provider_name == "mistral"
    assert result.original_exception is original


def test_http_validation_error_conversion() -> None:
    mock_response = httpx2.Response(status_code=422, content=b"Validation error")
    data = HTTPValidationErrorData()
    original = HTTPValidationError(data=data, raw_response=mock_response, body="Validation error")

    result = convert_exception(original, "mistral")

    assert isinstance(result, InvalidRequestError)
    assert result.provider_name == "mistral"
    assert result.original_exception is original


def test_response_validation_error_is_a_provider_error() -> None:
    """The message embeds the pydantic error's field paths, which must not pick the class."""

    class ChatCompletionResponse(BaseModel):
        auth_token: str
        rate_limit: int
        safety_settings: str

    with pytest.raises(ValidationError) as cause:
        ChatCompletionResponse.model_validate({})
    mock_response = httpx2.Response(status_code=200, content=b"{}")
    original = ResponseValidationError("Response validation failed", mock_response, cause.value, body="{}")

    result = convert_exception(original, "mistral")

    assert type(result) is ProviderError
    assert result.status_code == 200
    assert result.original_exception is original
