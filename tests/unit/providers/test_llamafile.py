from unittest.mock import patch

import pytest
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion import Choice as OpenAIChoice
from openai.types.chat.chat_completion_message import ChatCompletionMessage as OpenAIChatCompletionMessage

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.llamafile.llamafile import LlamafileProvider
from any_llm.providers.llamafile.utils import _convert_chat_completion
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import CompletionParams


@pytest.mark.asyncio
async def test_response_format_dict_raises() -> None:
    provider = LlamafileProvider()
    with pytest.raises(UnsupportedParameterError):
        await provider._acompletion(
            CompletionParams(
                model_id="llama3.1",
                messages=[{"role": "user", "content": "Hi"}],
                response_format={"type": "json_object"},
            )
        )


@pytest.mark.asyncio
async def test_calls_completion() -> None:
    provider = LlamafileProvider()
    params = CompletionParams(model_id="llama3.1", messages=[{"role": "user", "content": "Hi"}])
    sentinel = object()
    with patch.object(BaseOpenAIProvider, "_acompletion", autospec=True, return_value=sentinel) as mock_super:
        result = await provider._acompletion(params, temperature=0.1)
        assert result is sentinel
        mock_super.assert_called_once_with(provider, params, temperature=0.1)


@pytest.mark.asyncio
async def test_tools_raises() -> None:
    provider = LlamafileProvider()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "foo",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    with pytest.raises(UnsupportedParameterError):
        await provider._acompletion(
            CompletionParams(
                model_id="llama3.1",
                messages=[{"role": "user", "content": "Hi"}],
                tools=tools,
            )
        )


def test_convert_chat_completion_extracts_reasoning() -> None:
    openai_response = OpenAIChatCompletion(
        id="test-123",
        choices=[
            OpenAIChoice(
                finish_reason="stop",
                index=0,
                message=OpenAIChatCompletionMessage(
                    role="assistant",
                    content="<think>I need to greet the user</think>\n\nHello! How can I assist you?",
                ),
            )
        ],
        created=1234567890,
        model="llama-3.1",
        object="chat.completion",
    )

    result = _convert_chat_completion(openai_response)

    assert result.choices[0].message.reasoning is not None
    assert result.choices[0].message.reasoning.content == "I need to greet the user"
    assert result.choices[0].message.content == "Hello! How can I assist you?"


def test_convert_chat_completion_no_reasoning_tags() -> None:
    openai_response = OpenAIChatCompletion(
        id="test-456",
        choices=[
            OpenAIChoice(
                finish_reason="stop",
                index=0,
                message=OpenAIChatCompletionMessage(
                    role="assistant",
                    content="Just a simple response.",
                ),
            )
        ],
        created=1234567890,
        model="llama-3.1",
        object="chat.completion",
    )

    result = _convert_chat_completion(openai_response)

    assert result.choices[0].message.reasoning is None
    assert result.choices[0].message.content == "Just a simple response."
