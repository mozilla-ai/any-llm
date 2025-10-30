import pytest
from pydantic import BaseModel

from any_llm.providers.deepseek.utils import _preprocess_messages
from any_llm.types.completion import CompletionParams


class PersonResponseFormat(BaseModel):
    name: str
    age: int


@pytest.mark.asyncio
async def test_preprocess_messages_with_pydantic_model() -> None:
    """Test that Pydantic model is converted to DeepSeek JSON format."""
    messages = [{"role": "user", "content": "Generate a person"}]
    params = CompletionParams(
        model_id="deepseek-chat",
        messages=messages,
        response_format=PersonResponseFormat,
    )

    processed_params = _preprocess_messages(params)

    assert processed_params.response_format == {"type": "json_object"}

    # Should modify the user message to include JSON schema instructions
    assert len(processed_params.messages) == 1
    assert processed_params.messages[0]["role"] == "user"
    assert "JSON object" in processed_params.messages[0]["content"]
    assert "Generate a person" in processed_params.messages[0]["content"]


@pytest.mark.asyncio
async def test_preprocess_messages_without_response_format() -> None:
    """Test that messages are passed through unchanged when no response_format."""
    messages = [{"role": "user", "content": "Hello"}]
    params = CompletionParams(
        model_id="deepseek-chat",
        messages=messages,
        response_format=None,
    )

    processed_params = _preprocess_messages(params)

    assert processed_params.response_format is None
    assert processed_params.messages == messages


@pytest.mark.asyncio
async def test_preprocess_messages_with_non_pydantic_response_format() -> None:
    """Test that non-Pydantic response_format is passed through unchanged."""
    messages = [{"role": "user", "content": "Hello"}]
    response_format = {"type": "json_object"}
    params = CompletionParams(
        model_id="deepseek-chat",
        messages=messages,
        response_format=response_format,
    )

    processed_params = _preprocess_messages(params)
    assert processed_params.response_format == response_format
    assert processed_params.messages == messages
