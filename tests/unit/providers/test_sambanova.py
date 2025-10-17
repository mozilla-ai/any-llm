from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion import Choice as OpenAIChoice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice as OpenAIChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta as OpenAIChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage as OpenAIChatCompletionMessage

from any_llm.providers.sambanova.utils import (
    _convert_chat_completion,
    _convert_chat_completion_chunk,
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
        model="Meta-Llama-3.3-70B-Instruct",
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
        model="Meta-Llama-3.3-70B-Instruct",
        object="chat.completion",
    )

    result = _convert_chat_completion(openai_response)

    assert result.choices[0].message.reasoning is None
    assert result.choices[0].message.content == "Just a simple response."


def test_convert_chat_completion_multiple_reasoning_tags() -> None:
    openai_response = OpenAIChatCompletion(
        id="test-789",
        choices=[
            OpenAIChoice(
                finish_reason="stop",
                index=0,
                message=OpenAIChatCompletionMessage(
                    role="assistant",
                    content="<think>First thought</think>\n\n<think>Second thought</think>\n\nFinal answer.",
                ),
            )
        ],
        created=1234567890,
        model="Meta-Llama-3.3-70B-Instruct",
        object="chat.completion",
    )

    result = _convert_chat_completion(openai_response)

    assert result.choices[0].message.reasoning is not None
    assert result.choices[0].message.reasoning.content == "First thought\nSecond thought"
    assert result.choices[0].message.content == "Final answer."


def test_convert_chat_completion_chunk_extracts_reasoning() -> None:
    openai_chunk = OpenAIChatCompletionChunk(
        id="test-chunk-123",
        choices=[
            OpenAIChunkChoice(
                finish_reason=None,
                index=0,
                delta=OpenAIChoiceDelta(
                    role="assistant",
                    content="<think>Processing...</think>\n\nHere's my response",
                ),
            )
        ],
        created=1234567890,
        model="Meta-Llama-3.3-70B-Instruct",
        object="chat.completion.chunk",
    )

    result = _convert_chat_completion_chunk(openai_chunk)

    assert result.choices[0].delta.reasoning is not None
    assert result.choices[0].delta.reasoning.content == "Processing..."
    assert result.choices[0].delta.content == "Here's my response"


def test_convert_chat_completion_chunk_no_reasoning() -> None:
    openai_chunk = OpenAIChatCompletionChunk(
        id="test-chunk-456",
        choices=[
            OpenAIChunkChoice(
                finish_reason=None,
                index=0,
                delta=OpenAIChoiceDelta(
                    content="Just content",
                ),
            )
        ],
        created=1234567890,
        model="Meta-Llama-3.3-70B-Instruct",
        object="chat.completion.chunk",
    )

    result = _convert_chat_completion_chunk(openai_chunk)

    assert result.choices[0].delta.reasoning is None
    assert result.choices[0].delta.content == "Just content"


def test_convert_chat_completion_handles_multiline_reasoning() -> None:
    openai_response = OpenAIChatCompletion(
        id="test-multiline",
        choices=[
            OpenAIChoice(
                finish_reason="stop",
                index=0,
                message=OpenAIChatCompletionMessage(
                    role="assistant",
                    content="<think>\nMultiline\nreasoning\nprocess\n</think>\n\nFinal answer.",
                ),
            )
        ],
        created=1234567890,
        model="Meta-Llama-3.3-70B-Instruct",
        object="chat.completion",
    )

    result = _convert_chat_completion(openai_response)

    assert result.choices[0].message.reasoning is not None
    assert result.choices[0].message.reasoning.content == "\nMultiline\nreasoning\nprocess\n"
    assert result.choices[0].message.content == "Final answer."
