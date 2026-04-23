import pytest
from pydantic import BaseModel, ValidationError

from any_llm.exceptions import GatewayTimeoutError, InsufficientFundsError, UpstreamProviderError
from any_llm.utils.exception_handler import _handle_exception, convert_exception


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
