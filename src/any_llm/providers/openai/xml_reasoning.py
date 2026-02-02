"""XML Reasoning parsing support for OpenAI-compatible providers.

This module provides an intermediate base class and helper functions for providers
that need to extract reasoning content from XML tags (e.g., <think>, <thinking>)
rather than using native reasoning fields.
"""

from collections.abc import AsyncIterator

from openai._streaming import AsyncStream
from openai.types.chat.chat_completion import ChatCompletion as OpenAIChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk as OpenAIChatCompletionChunk
from typing_extensions import override

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, Reasoning
from any_llm.utils.reasoning import process_streaming_reasoning_chunks


def get_chunk_content(chunk: ChatCompletionChunk) -> str | None:
    """Extract content from a ChatCompletionChunk."""
    return chunk.choices[0].delta.content if len(chunk.choices) > 0 else None


def set_chunk_content(chunk: ChatCompletionChunk, content: str | None) -> ChatCompletionChunk:
    """Set content on a ChatCompletionChunk."""
    chunk.choices[0].delta.content = content
    return chunk


def set_chunk_reasoning(chunk: ChatCompletionChunk, reasoning: str) -> ChatCompletionChunk:
    """Set reasoning on a ChatCompletionChunk."""
    chunk.choices[0].delta.reasoning = Reasoning(content=reasoning)
    return chunk


def wrap_chunks_with_xml_reasoning(
    chunks: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[ChatCompletionChunk]:
    """Wrap a chunk iterator to extract reasoning from XML tags.

    This helper is useful for providers that don't inherit from XMLReasoningOpenAIProvider
    but still need XML reasoning extraction (e.g., HuggingFace which has a different base class).

    Args:
        chunks: Async iterator of ChatCompletionChunk objects

    Returns:
        Async iterator with XML reasoning tags extracted and converted to Reasoning objects
    """
    return process_streaming_reasoning_chunks(
        chunks,
        get_content=get_chunk_content,
        set_content=set_chunk_content,
        set_reasoning=set_chunk_reasoning,
    )


class XMLReasoningOpenAIProvider(BaseOpenAIProvider):
    """Base provider for OpenAI-compatible services that need XML reasoning extraction.

    This class extends BaseOpenAIProvider to add XML reasoning tag parsing for
    providers whose upstream APIs embed reasoning in XML tags (e.g., <think>, <thinking>)
    rather than using native reasoning fields.

    Providers that need this behavior should inherit from this class instead of
    BaseOpenAIProvider. They will automatically get:
    - Streaming: XML tags extracted and converted to Reasoning objects in real-time
    - Non-streaming: Use the shared utils in xml_reasoning_utils.py for conversion

    Subclasses should still override `_convert_completion_response` and
    `_convert_completion_chunk_response` to use the XML-aware conversion utilities.
    """

    SUPPORTS_COMPLETION_REASONING = True

    @override
    def _convert_completion_response_async(
        self, response: OpenAIChatCompletion | AsyncStream[OpenAIChatCompletionChunk]
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        """Convert completion response with XML reasoning extraction for streaming."""
        if isinstance(response, OpenAIChatCompletion):
            return self._convert_completion_response(response)

        async def chunk_iterator() -> AsyncIterator[ChatCompletionChunk]:
            async for chunk in response:
                yield self._convert_completion_chunk_response(chunk)

        return wrap_chunks_with_xml_reasoning(chunk_iterator())
