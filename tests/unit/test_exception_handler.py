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
from any_llm.utils.exception_handler import (
    _ERROR_PATTERNS,
    _STATUS_ERROR_CLASSES,
    _handle_exception,
    convert_exception,
    handle_exceptions,
)


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


@pytest.mark.parametrize("value", [True, 30.5, object()])
def test_convert_exception_rejects_unusable_retry_after_values(value: object) -> None:
    """Only a string or a plain int is a usable retry hint.

    bool is excluded deliberately even though it is an int subclass: ``True``
    would stringify to ``"True"``, which is not a delay a caller can act on.
    """
    error = _ResponseStatusError(429, "Rate limit exceeded")
    error.response.headers = {"retry-after": value}  # type: ignore[attr-defined]
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


def test_every_classified_class_accepts_the_metadata_keywords() -> None:
    """Every class either table can select must accept the structured metadata keywords.

    convert_exception constructs through a ``type[AnyLLMError]`` variable, so
    mypy checks the call against the base signature only. A subclass with an
    incompatible ``__init__`` (MissingApiKeyError, BatchNotCompleteError) would
    pass the type check and raise TypeError inside the exception handler,
    masking the provider's original error. Both the message table and the status
    table feed that construction site, so both are covered here.
    """
    classified = {error_class for _pattern, error_class in _ERROR_PATTERNS}
    classified.update(_STATUS_ERROR_CLASSES.values())
    classified.update({InvalidRequestError, ProviderError})
    for error_class in classified:
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


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, AuthenticationError),
        (402, InsufficientFundsError),
        (403, AuthenticationError),
        (404, ModelNotFoundError),
        (429, RateLimitError),
        (502, UpstreamProviderError),
        (504, GatewayTimeoutError),
    ],
)
def test_unambiguous_status_wins_over_an_unhelpful_message(status_code: int, expected: type[AnyLLMError]) -> None:
    """A provider that returns a generic body still classifies from its status.

    Before, "Something went wrong" matched no pattern and fell through to
    ProviderError regardless of the status, so a 429 or a 401 was indistinguishable
    from an outage.
    """
    assert type(convert_exception(_StatusError(status_code, "Something went wrong"), "openai")) is expected


@pytest.mark.parametrize("status_code", [401, 402, 404, 429, 502, 504])
def test_unambiguous_status_wins_over_a_contradicting_message(status_code: int) -> None:
    """The status is what the provider sent; a message coincidence cannot override it."""
    error = _StatusError(status_code, "The response was filtered due to the content policy")
    assert not isinstance(convert_exception(error, "openai"), ContentFilterError)


@pytest.mark.parametrize("status_code", [500, 503, 529])
def test_server_errors_are_provider_errors_regardless_of_message(status_code: int) -> None:
    """The misclassification called out in #1190: a 5xx whose body mentions
    "policy" is a provider fault, not a content-filter rejection."""
    error = _StatusError(status_code, "Internal error while applying the content policy")
    assert type(convert_exception(error, "openai")) is ProviderError


@pytest.mark.parametrize("status_code", [400, 422])
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("This model's maximum context length is 8192 tokens", ContextLengthExceededError),
        ("The response was filtered due to the content policy", ContentFilterError),
        ("Incorrect API key provided", AuthenticationError),
        ("The model gpt-9 does not exist", ModelNotFoundError),
        ("Budget exceeded for this project", InsufficientFundsError),
    ],
)
def test_rejected_request_keeps_the_specific_cause_from_the_message(
    status_code: int, message: str, expected: type[AnyLLMError]
) -> None:
    """400/422 says the request was rejected but not why.

    ContextLengthExceededError and ContentFilterError have no status of their own
    (OpenAI returns 400 for both) and some providers report a bad key as a 400, so
    the patterns still decide here. Keying 400 straight to InvalidRequestError would
    make those classes unreachable for any provider that reports a status.
    """
    assert type(convert_exception(_StatusError(status_code, message), "openai")) is expected


@pytest.mark.parametrize("status_code", [400, 422])
def test_rejected_request_falls_back_to_invalid_request(status_code: int) -> None:
    """With no specific cause in the message, a rejected request is the caller's
    problem, so it lands on InvalidRequestError rather than ProviderError."""
    error = _StatusError(status_code, "Something went wrong")
    assert type(convert_exception(error, "openai")) is InvalidRequestError


@pytest.mark.parametrize("status_code", [409, 413, 418, 302])
def test_unmapped_status_defers_to_the_message_patterns(status_code: int) -> None:
    """A status with no agreed meaning here leaves the patterns in charge."""
    error = _StatusError(status_code, "This model's maximum context length is 8192 tokens")
    assert type(convert_exception(error, "openai")) is ContextLengthExceededError


def test_status_free_failures_still_use_the_message_patterns() -> None:
    """The status-free path is unchanged, so a timeout or an SDK error carrying no
    status classifies exactly as it did before."""
    assert type(convert_exception(Exception("Rate limit reached"), "openai")) is RateLimitError
    assert type(convert_exception(TimeoutError("request timed out"), "openai")) is ProviderError


class _SdkStream:
    """An SDK stream like openai's AsyncStream: iterable, closed via close(), no aclose()."""

    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self) -> Any:
        for item in range(3):
            yield item

    async def close(self) -> None:
        self.closed = True


_STREAM_ERROR = "stream broke"
_CLOSE_ERROR = "close broke"


class _SyncCloseStream:
    """An iterable whose close() is synchronous and fails."""

    async def __aiter__(self) -> Any:
        yield 1
        raise ValueError(_STREAM_ERROR)

    def close(self) -> None:
        raise RuntimeError(_CLOSE_ERROR)


class _Provider:
    PROVIDER_NAME = "test"

    def __init__(self) -> None:
        self.generator_closed = False
        self.sdk_stream = _SdkStream()

    @handle_exceptions(wrap_streaming=True)
    async def stream_generator(self) -> Any:
        async def generate() -> Any:
            try:
                for item in range(3):
                    yield item
            finally:
                self.generator_closed = True

        return generate()

    @handle_exceptions(wrap_streaming=True)
    async def stream_sdk(self) -> Any:
        return self.sdk_stream

    @handle_exceptions(wrap_streaming=True)
    async def stream_sync_close(self) -> Any:
        return _SyncCloseStream()

    @handle_exceptions(wrap_streaming=True)
    async def stream_plain(self) -> Any:
        async def generate() -> Any:
            yield 1

        return generate().__aiter__()


@pytest.mark.asyncio
async def test_closing_the_wrapped_stream_closes_the_provider_stream() -> None:
    """A caller's aclose() must reach the provider stream, whether it spells it aclose or close."""
    provider = _Provider()

    wrapped = await provider.stream_generator()
    await wrapped.__anext__()
    await wrapped.aclose()
    assert provider.generator_closed

    wrapped = await provider.stream_sdk()
    await wrapped.__anext__()
    await wrapped.aclose()
    assert provider.sdk_stream.closed


@pytest.mark.asyncio
async def test_provider_stream_closes_on_exhaustion_and_iteration_errors() -> None:
    """Cleanup runs on every exit: exhaustion, an iteration error (still surfaced), a sync closer that fails."""
    provider = _Provider()

    assert [item async for item in await provider.stream_generator()] == [0, 1, 2]
    assert provider.generator_closed

    with pytest.raises(ValueError, match=_STREAM_ERROR):  # the stream's error wins over the failing closer
        async for _ in await provider.stream_sync_close():
            pass

    assert [item async for item in await provider.stream_plain()] == [1]  # no close method at all is fine
