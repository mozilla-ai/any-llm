import pytest
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion

from any_llm.exceptions import ProviderError
from any_llm.providers.openai.utils import _convert_chat_completion


def test_convert_chat_completion_with_empty_response() -> None:
    # Simulating the malformed response described in the PR
    # ChatCompletion(id=None, choices=None, created=None, model=None, object='chat.completion', ...)
    openai_response = OpenAIChatCompletion.model_construct(
        id=None,
        choices=None, # type: ignore
        created=None, # type: ignore
        model=None,   # type: ignore
        object="chat.completion",
    )

    with pytest.raises(ProviderError) as exc_info:
        _convert_chat_completion(openai_response)

    assert "Provider returned an empty response" in str(exc_info.value)

def test_convert_chat_completion_with_partial_none_response() -> None:
    # If not all THREE (id, choices, model) are None, it should NOT raise ProviderError early.
    # It might fail later if other required fields like 'created' are missing or invalid,
    # but the specific guard being tested here should not trigger.
    openai_response = OpenAIChatCompletion.model_construct(
        id="test-id",
        choices=None, # type: ignore
        created=1234567890,
        model=None,   # type: ignore
        object="chat.completion",
    )

    # In this case, it will fail later during ChatCompletion.model_validate(normalized)
    # because 'choices' is None, or during _normalize_openai_dict_response if it expect choices to be a list.
    
    # Actually _normalize_openai_dict_response handles choices=None:
    # choices = response_dict.get("choices")
    # if isinstance(choices, list): ...
    
    # But ChatCompletion.model_validate(normalized) will fail because choices is required.
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        _convert_chat_completion(openai_response)