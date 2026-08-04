from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError,
    ContentFilterError,
    ContextLengthExceededError,
    GatewayTimeoutError,
    InsufficientFundsError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    UpstreamProviderError,
)
from any_llm.utils.exception_handler import _ERROR_PATTERNS, _handle_exception, convert_exception


class _StatusError(Exception):
    """Provider error exposing an HTTP status directly, like openai's APIStatusError."""

    def __init__(self, status_code: int, message: str = "boom", **fields: Any) -> None:
        super().__init__(message)
        self.status_code = status_code
        for name, value in fields.items():
            setattr(self, name, value)


class _ResponseStatusError(Exception):
    """Provider error carrying the status only on an attached response object."""

    def __init__(self, status_code: int, message: str = "boom") -> None:
        super().__init__(message)

        class _Response:
            pass

        response = _Response()
        response.status_code = status_code  # type: ignore[attr-defined]
        self.response = response


class _BodyError(Exception):
    """Provider error carrying structured fields only in a response body."""

    def __init__(self, body: Any, message: str = "boom") -> None:
        super().__init__(message)
        self.body = body


def test_validation_error_bubbles_up_unchanged() -> None:
    """Test that pydantic ValidationError bubbles up unchanged.

    When using response_format with a Pydantic model, the SDK validates the response
    internally. If the LLM produces output that doesn't conform to the schema, pydantic
    raises ValidationError. This should bubble up unchanged.
    See: https://github.com/mozilla-ai/any-llm/issues/799
    """

    class Sample(BaseModel):
        value: str

    with pytest.raises(ValidationError) as exc_info:
        Sample.model_validate({"value": 123})

    original = exc_info.value

    with pytest.raises(ValidationError) as raised:
        _handle_exception(original, "openai")

    assert raised.value is original


def test_convert_exception_insufficient_funds() -> None:
    original = Exception("Insufficient funds for this request")
    result = convert_exception(original, "gateway")
    assert isinstance(result, InsufficientFundsError)
    assert result.provider_name == "gateway"
    assert result.original_exception is original


def test_convert_exception_payment_required() -> None:
    original = Exception("Payment required")
    result = convert_exception(original, "gateway")
    assert isinstance(result, InsufficientFundsError)


def test_convert_exception_budget_exceeded() -> None:
    original = Exception("Budget exceeded for project")
    result = convert_exception(original, "gateway")
    assert isinstance(result, InsufficientFundsError)


def test_convert_exception_bad_gateway() -> None:
    original = Exception("Bad gateway")
    result = convert_exception(original, "gateway")
    assert isinstance(result, UpstreamProviderError)
    assert result.provider_name == "gateway"
    assert result.original_exception is original


def test_convert_exception_upstream_provider_error() -> None:
    original = Exception("Upstream provider error")
    result = convert_exception(original, "gateway")
    assert isinstance(result, UpstreamProviderError)


def test_convert_exception_gateway_timeout() -> None:
    original = Exception("Gateway timeout")
    result = convert_exception(original, "gateway")
    assert isinstance(result, GatewayTimeoutError)
    assert result.provider_name == "gateway"
    assert result.original_exception is original


def test_convert_exception_api_key_is_invalid() -> None:
    """Test that 'Your API key is invalid' is classified as AuthenticationError.

    When a provider returns a message like 'Your API key is invalid', the word
    'invalid' appears after 'api key' with intervening words. The auth regex must
    catch this before the broad 'invalid' pattern misclassifies it as InvalidRequestError.
    """
    from any_llm.exceptions import AuthenticationError, InvalidRequestError

    original = Exception("Your API key is invalid")
    result = convert_exception(original, "openai")
    assert isinstance(result, AuthenticationError)
    assert not isinstance(result, InvalidRequestError)


def test_convert_exception_api_key_not_valid_with_intervening_words() -> None:
    """Test that variations with words between 'api key' and 'invalid' are caught."""
    from any_llm.exceptions import AuthenticationError

    original = Exception("The provided api key seems invalid for this request")
    result = convert_exception(original, "openai")
    assert isinstance(result, AuthenticationError)


def test_convert_exception_invalid_api_key_word_boundary() -> None:
    """Test that 'invalid' near 'api key' is treated as auth error, not invalid request."""
    from any_llm.exceptions import AuthenticationError

    original = Exception("Invalid API key provided")
    result = convert_exception(original, "openai")
    assert isinstance(result, AuthenticationError)


def test_any_llm_error_metadata_defaults_to_none() -> None:
    """An exception constructed without metadata still exposes the fields."""
    error = AnyLLMError("something failed")
    assert error.status_code is None
    assert error.code is None
    assert error.param is None
    assert error.error_type is None


def test_convert_exception_carries_status_code_from_attribute() -> None:
    result = convert_exception(_StatusError(400, "Invalid request"), "openai")
    assert isinstance(result, InvalidRequestError)
    assert result.status_code == 400


def test_convert_exception_carries_status_code_from_response() -> None:
    """A status carried only on the attached response is still recovered."""
    result = convert_exception(_ResponseStatusError(400, "Invalid request"), "openai")
    assert result.status_code == 400


def test_convert_exception_prefers_status_code_attribute_over_response() -> None:
    error = _ResponseStatusError(500, "Invalid request")
    error.status_code = 400  # type: ignore[attr-defined]
    assert convert_exception(error, "openai").status_code == 400


def test_convert_exception_ignores_non_integer_status_code() -> None:
    """A provider that stores something other than an int is treated as absent."""
    result = convert_exception(_StatusError("400", "Invalid request"), "openai")  # type: ignore[arg-type]
    assert result.status_code is None


def test_convert_exception_ignores_response_without_usable_status() -> None:
    """An attached response that reports no integer status yields no status_code."""
    error = _ResponseStatusError(400, "Invalid request")
    error.response.status_code = None  # type: ignore[attr-defined]
    assert convert_exception(error, "openai").status_code is None


def test_convert_exception_carries_param_code_and_type_from_attributes() -> None:
    """The OpenAI SDK populates these directly, so they are read off the exception."""
    error = _StatusError(
        400,
        "Invalid request",
        param="reasoning_effort",
        code="unsupported_parameter",
        type="invalid_request_error",
    )
    result = convert_exception(error, "openai")
    assert result.param == "reasoning_effort"
    assert result.code == "unsupported_parameter"
    assert result.error_type == "invalid_request_error"


def test_convert_exception_reads_metadata_from_flat_body() -> None:
    error = _BodyError(
        {"param": "reasoning_effort", "code": "unsupported_parameter", "type": "invalid_request_error"},
        "Invalid request",
    )
    result = convert_exception(error, "openai")
    assert result.param == "reasoning_effort"
    assert result.code == "unsupported_parameter"
    assert result.error_type == "invalid_request_error"


def test_convert_exception_reads_metadata_from_nested_error_body() -> None:
    """Bodies that keep the ``{"error": {...}}`` nesting are unwrapped."""
    error = _BodyError(
        {"error": {"param": "reasoning_effort", "code": "unsupported_parameter", "type": "invalid_request_error"}},
        "Invalid request",
    )
    result = convert_exception(error, "openai")
    assert result.param == "reasoning_effort"
    assert result.code == "unsupported_parameter"
    assert result.error_type == "invalid_request_error"


def test_convert_exception_prefers_attribute_over_body() -> None:
    error = _BodyError({"param": "from_body"}, "Invalid request")
    error.param = "from_attribute"  # type: ignore[attr-defined]
    assert convert_exception(error, "openai").param == "from_attribute"


def test_convert_exception_prefers_nested_error_detail_over_envelope() -> None:
    """When a body nests an ``error`` dict, that detail wins over the envelope."""
    error = _BodyError({"param": "envelope", "error": {"param": "detail"}}, "Invalid request")
    assert convert_exception(error, "openai").param == "detail"


def test_convert_exception_ignores_anthropic_envelope_type() -> None:
    """Anthropic's envelope carries a literal ``"type": "error"`` alongside the
    real category in the nested dict; the nested value must win."""
    error = _BodyError(
        {"type": "error", "error": {"type": "invalid_request_error", "message": "Invalid request"}},
        "Invalid request",
    )
    assert convert_exception(error, "openai").error_type == "invalid_request_error"


@pytest.mark.parametrize("body", [None, "not a dict", 42, ["param"], {"error": "not a dict"}])
def test_convert_exception_tolerates_unusable_body(body: Any) -> None:
    """A body that is not a dict (or whose ``error`` is not a dict) yields no metadata."""
    result = convert_exception(_BodyError(body, "Invalid request"), "openai")
    assert result.param is None
    assert result.code is None
    assert result.error_type is None


def test_convert_exception_ignores_non_string_body_values() -> None:
    """OpenAI sends ``'code': None`` on some errors; non-strings are treated as absent."""
    error = _BodyError({"param": None, "code": 500, "type": ["invalid"]}, "Invalid request")
    result = convert_exception(error, "openai")
    assert result.param is None
    assert result.code is None
    assert result.error_type is None


def test_convert_exception_leaves_metadata_none_for_non_http_failure() -> None:
    """A timeout carries no HTTP status, so every structured field stays None."""
    result = convert_exception(TimeoutError("request timed out"), "openai")
    assert isinstance(result, ProviderError)
    assert result.status_code is None
    assert result.code is None
    assert result.param is None
    assert result.error_type is None


def test_convert_exception_carries_retry_after_from_response_headers() -> None:
    """RateLimitError.retry_after is filled from the Retry-After header."""
    error = _ResponseStatusError(429, "Rate limit exceeded")
    error.response.headers = {"retry-after": "30"}  # type: ignore[attr-defined]
    result = convert_exception(error, "openai")
    assert isinstance(result, RateLimitError)
    assert result.retry_after == "30"


def test_convert_exception_reads_retry_after_case_insensitively() -> None:
    """A plain dict is case-sensitive, so the canonical header spelling is tried too."""
    error = _ResponseStatusError(429, "Rate limit exceeded")
    error.response.headers = {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}  # type: ignore[attr-defined]
    result = convert_exception(error, "openai")
    assert isinstance(result, RateLimitError)
    assert result.retry_after == "Wed, 21 Oct 2026 07:28:00 GMT"


def test_convert_exception_stringifies_numeric_retry_after() -> None:
    """A header store holding an int still yields the declared str type."""
    error = _ResponseStatusError(429, "Rate limit exceeded")
    error.response.headers = {"retry-after": 30}  # type: ignore[attr-defined]
    result = convert_exception(error, "openai")
    assert isinstance(result, RateLimitError)
    assert result.retry_after == "30"


def test_convert_exception_retry_after_is_none_without_the_header() -> None:
    error = _ResponseStatusError(429, "Rate limit exceeded")
    error.response.headers = {}  # type: ignore[attr-defined]
    result = convert_exception(error, "openai")
    assert isinstance(result, RateLimitError)
    assert result.retry_after is None


def test_convert_exception_retry_after_is_none_without_usable_headers() -> None:
    """A response with no headers mapping at all does not raise."""
    error = _ResponseStatusError(429, "Rate limit exceeded")
    result = convert_exception(error, "openai")
    assert isinstance(result, RateLimitError)
    assert result.retry_after is None


def test_convert_exception_ignores_retry_after_on_non_rate_limit_errors() -> None:
    """Only RateLimitError declares retry_after, so nothing else grows the field."""
    error = _ResponseStatusError(400, "Invalid request")
    error.response.headers = {"retry-after": "30"}  # type: ignore[attr-defined]
    result = convert_exception(error, "openai")
    assert isinstance(result, InvalidRequestError)
    assert not hasattr(result, "retry_after")


def test_convert_exception_carries_metadata_onto_rate_limit_error() -> None:
    """RateLimitError overrides ``__init__``, so it must forward the new fields."""
    error = _StatusError(429, "Rate limit exceeded", code="rate_limit_exceeded", type="rate_limit_error")
    result = convert_exception(error, "openai")
    assert isinstance(result, RateLimitError)
    assert result.status_code == 429
    assert result.code == "rate_limit_exceeded"
    assert result.error_type == "rate_limit_error"
    assert result.retry_after is None


def test_convert_exception_preserves_already_unified_error_untouched() -> None:
    """An AnyLLMError passes through with the metadata it was built with."""
    original = InvalidRequestError("Invalid request", status_code=400, param="reasoning_effort")
    result = convert_exception(original, "openai")
    assert result is original
    assert result.status_code == 400
    assert result.param == "reasoning_effort"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Rate limit exceeded", RateLimitError),
        ("Unauthorized", AuthenticationError),
        ("This model's maximum context length is 8192 tokens", ContextLengthExceededError),
        ("The model gpt-9 does not exist", ModelNotFoundError),
        ("Your request was blocked by the content policy", ContentFilterError),
        ("Invalid value for temperature", InvalidRequestError),
        ("Payment required", InsufficientFundsError),
        ("Bad gateway", UpstreamProviderError),
        ("Gateway timeout", GatewayTimeoutError),
        ("Connection reset by peer", ProviderError),
        ("Wibble wobble", ProviderError),
    ],
)
def test_convert_exception_classification_order_is_pinned(message: str, expected: type[AnyLLMError]) -> None:
    """One representative message per entry in the ordered pattern table.

    The table is matched first-hit-wins, so a reordering or a broadened pattern
    silently re-routes failures to a different exception type. Pinning every
    entry (including the unmatched fallback) makes that visible.
    """
    assert type(convert_exception(Exception(message), "openai")) is expected


def test_every_pattern_class_accepts_the_metadata_keywords() -> None:
    """Every class in the table must accept the structured metadata keywords.

    convert_exception constructs through a ``type[AnyLLMError]`` variable, so
    mypy checks the call against the base signature only. A subclass with an
    incompatible ``__init__`` (MissingApiKeyError, BatchNotCompleteError) would
    pass the type check and raise TypeError inside the exception handler,
    masking the provider's original error.
    """
    for _pattern, error_class in _ERROR_PATTERNS:
        error = error_class(
            message="boom",
            original_exception=ValueError("boom"),
            provider_name="openai",
            status_code=400,
            code="unsupported_parameter",
            param="reasoning_effort",
            error_type="invalid_request_error",
        )
        assert error.status_code == 400
        assert error.code == "unsupported_parameter"
        assert error.param == "reasoning_effort"
        assert error.error_type == "invalid_request_error"
