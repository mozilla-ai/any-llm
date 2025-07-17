from typing import Any, Iterator, TypeVar
from openai._streaming import Stream as OpenAIStream
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion, Choice as OpenAIChoice
from openai.types.completion_usage import CompletionUsage as OpenAICompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage as OpenAIChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall as OpenAIChatCompletionMessageToolCall,
    Function as OpenAIFunction,
)


class ChatCompletion(OpenAIChatCompletion):
    """AnyLLM ChatCompletion type, identical to OpenAI's ChatCompletion."""


class ChatCompletionChunk(OpenAIChatCompletionChunk):
    """AnyLLM ChatCompletionChunk type, identical to OpenAI's ChatCompletionChunk."""


class Choice(OpenAIChoice):
    """AnyLLM Choice type, identical to OpenAI's Choice."""


class CompletionUsage(OpenAICompletionUsage):
    """AnyLLM CompletionUsage type, identical to OpenAI's CompletionUsage."""


class ChatCompletionMessage(OpenAIChatCompletionMessage):
    """AnyLLM ChatCompletionMessage type, identical to OpenAI's ChatCompletionMessage."""


class ChatCompletionMessageToolCall(OpenAIChatCompletionMessageToolCall):
    """AnyLLM ChatCompletionMessageToolCall type, identical to OpenAI's ChatCompletionMessageToolCall."""


class Function(OpenAIFunction):
    """AnyLLM Function type, identical to OpenAI's Function."""


_T = TypeVar("_T")


class Stream(OpenAIStream[_T]):
    """Custom stream wrapper that converts OpenAI stream chunks to AnyLLM ChatCompletionChunk types."""

    def __init__(self, openai_stream: OpenAIStream[Any]):
        self._openai_stream = openai_stream

    def __iter__(self) -> Iterator[_T]:
        for chunk in self._openai_stream:
            # Convert OpenAI chunk to AnyLLM ChatCompletionChunk
            assert isinstance(chunk, OpenAIChatCompletionChunk), f"Expected ChatCompletionChunk, got {type(chunk)}"
            yield ChatCompletionChunk.model_validate(chunk.model_dump())

    def __next__(self) -> _T:
        chunk = next(self._openai_stream)
        assert isinstance(chunk, OpenAIChatCompletionChunk), f"Expected ChatCompletionChunk, got {type(chunk)}"
        return ChatCompletionChunk.model_validate(chunk.model_dump())

    def close(self) -> None:
        """Close the underlying stream."""
        if hasattr(self._openai_stream, "close"):
            self._openai_stream.close()

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


__all__ = [
    "ChatCompletion",
    "ChatCompletionChunk",
    "Choice",
    "CompletionUsage",
    "ChatCompletionMessage",
    "ChatCompletionMessageToolCall",
    "Function",
    "Stream",
]
