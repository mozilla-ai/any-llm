import json
from typing import Any

import pytest

from any_llm.exceptions import (
    AuthenticationError,
    ContentFilterError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
)
from any_llm.utils.exception_handler import _handle_exception, convert_exception


@pytest.mark.parametrize("provider", ["openai", "anthropic", "groq", "together", "cerebras", "mistral"])
@pytest.mark.parametrize("field", ["result", "auth_token", "rate_limit", "safety_settings"])
def test_sdk_response_validation_is_a_provider_error(provider: str, field: str) -> None:
    module_name = {"cerebras": "cerebras.cloud.sdk", "mistral": "mistralai.client.errors"}.get(provider, provider)
    sdk = pytest.importorskip(module_name)
    http = pytest.importorskip("httpx2" if provider in {"openai", "mistral"} else "httpx")
    response = http.Response(200, request=http.Request("POST", "https://example.test/completions"))
    message = f"Data returned by API invalid for expected schema: {field}"
    if provider == "mistral":
        original = sdk.ResponseValidationError(message, response, ValueError("bad response"), body="{}")
    else:
        original = sdk.APIResponseValidationError(response=response, body={}, message=message)

    with pytest.raises(ProviderError) as raised:
        _handle_exception(original, provider, unified_exceptions=True)

    assert type(raised.value) is ProviderError
    assert raised.value.original_exception is original
    assert raised.value.status_code == 200
    assert raised.value.provider_name == provider
    assert raised.value.__cause__ is original


@pytest.mark.parametrize("parameter", ["not_a_param", "auth_token", "safety_settings", "rate_limit"])
def test_unexpected_keyword_is_a_caller_error(parameter: str) -> None:
    def invoke(*, model: str) -> None:
        pass

    with pytest.raises(TypeError) as original:
        invoke(model="test", **{parameter: 1})

    result = convert_exception(original.value, "openai")

    assert type(result) is InvalidRequestError
    assert result.original_exception is original.value
    assert result.status_code is None


def test_duplicate_argument_is_a_caller_error() -> None:
    def invoke(auth_token: int) -> None:
        pass

    call: Any = invoke
    with pytest.raises(TypeError) as original:
        call(1, auth_token=2)

    assert type(convert_exception(original.value, "anthropic")) is InvalidRequestError


@pytest.mark.parametrize("keyword_only", [False, True])
def test_missing_required_arguments_are_caller_errors(keyword_only: bool) -> None:
    def positional(auth_token: str, safety_settings: str) -> None:
        pass

    def keyword(*, auth_token: str, safety_settings: str) -> None:
        pass

    invoke: Any = keyword if keyword_only else positional
    with pytest.raises(TypeError) as original:
        invoke()

    assert type(convert_exception(original.value, "mistral")) is InvalidRequestError


@pytest.mark.parametrize("optional", [False, True])
def test_excess_positional_arguments_are_caller_errors(optional: bool) -> None:
    def fixed(auth_token: str) -> None:
        pass

    def with_default(auth_token: int = 0) -> None:
        pass

    invoke: Any = with_default if optional else fixed
    with pytest.raises(TypeError) as original:
        invoke("one", "two")

    assert type(convert_exception(original.value, "openai")) is InvalidRequestError


def test_json_decode_error_is_not_blanket_classified_as_a_caller_error() -> None:
    with pytest.raises(json.JSONDecodeError) as original:
        json.loads("not JSON")

    assert type(convert_exception(original.value, "openai")) is ProviderError


def test_unified_exceptions_disabled_preserves_signature_error() -> None:
    original = TypeError("create() got an unexpected keyword argument 'auth_token'")
    with pytest.warns(DeprecationWarning, match="Provider-specific exceptions"), pytest.raises(TypeError) as raised:
        _handle_exception(original, "openai", unified_exceptions=False)

    assert raised.value is original


@pytest.mark.parametrize("message", ["rate limit exceeded", "invalid api key", "content safety blocked"])
def test_stream_error_with_status_200_keeps_message_classification(message: str) -> None:
    sdk = pytest.importorskip("anthropic")
    http = pytest.importorskip("httpx")
    response = http.Response(200, request=http.Request("POST", "https://example.test/messages"))
    original = sdk.APIStatusError(message, response=response, body={})
    expected = {
        "rate limit exceeded": RateLimitError,
        "invalid api key": AuthenticationError,
        "content safety blocked": ContentFilterError,
    }[message]

    assert type(convert_exception(original, "anthropic")) is expected
