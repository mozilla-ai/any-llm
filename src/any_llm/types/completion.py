from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion import Choice as OpenAIChoice
from openai.types.chat.chat_completion_message import ChatCompletionMessage as OpenAIChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall as OpenAIChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall as OpenAIChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import Function as OpenAIFunction
from openai.types.completion_usage import CompletionUsage as OpenAICompletionUsage
from openai.types import CreateEmbeddingResponse as OpenAICreateEmbeddingResponse
from openai.types.embedding import Embedding as OpenAIEmbedding
from openai.types.create_embedding_response import Usage as OpenAIUsage
from openai.types.chat.chat_completion_chunk import Choice as OpenAIChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta as OpenAIChoiceDelta
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall as OpenAIChoiceDeltaToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCallFunction as OpenAIChoiceDeltaToolCallFunction

from pydantic import BaseModel

# See https://github.com/mozilla-ai/any-llm/issues/95:
# OpenAI Completion API doesn't include reasoning information, so we need to extend the openai type


class Reasoning(BaseModel):
    content: str


class ChatCompletionMessage(OpenAIChatCompletionMessage):
    reasoning: Reasoning | None = None


class Choice(OpenAIChoice):
    message: ChatCompletionMessage


class ChatCompletion(OpenAIChatCompletion):
    choices: list[Choice]  # type: ignore[assignment]


class ChoiceDelta(OpenAIChoiceDelta):
    reasoning: Reasoning | None = None


class ChunkChoice(OpenAIChunkChoice):
    delta: ChoiceDelta


class ChatCompletionChunk(OpenAIChatCompletionChunk):
    choices: list[ChunkChoice]  # type: ignore[assignment]


ChatCompletionMessageToolCall = OpenAIChatCompletionMessageToolCall
Function = OpenAIFunction
CompletionUsage = OpenAICompletionUsage
CreateEmbeddingResponse = OpenAICreateEmbeddingResponse
Embedding = OpenAIEmbedding
Usage = OpenAIUsage
ChoiceDeltaToolCall = OpenAIChoiceDeltaToolCall
ChoiceDeltaToolCallFunction = OpenAIChoiceDeltaToolCallFunction


ChatCompletionMessageFunctionToolCall = OpenAIChatCompletionMessageFunctionToolCall
