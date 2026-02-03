import pytest
from pydantic import BaseModel, ValidationError

from any_llm.utils.exception_handler import _handle_exception


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
