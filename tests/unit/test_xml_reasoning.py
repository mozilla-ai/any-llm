from collections.abc import AsyncIterator

import pytest

from any_llm.providers.openai.xml_reasoning import (
    get_chunk_content,
    set_chunk_content,
    set_chunk_reasoning,
    wrap_chunks_with_xml_reasoning,
)
from any_llm.types.completion import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChunkChoice,
    Reasoning,
)


def _make_chunk(
    content: str | None = None,
    reasoning: Reasoning | None = None,
) -> ChatCompletionChunk:
    """Create a minimal ChatCompletionChunk for testing."""
    return ChatCompletionChunk(
        id="test-chunk",
        choices=[
            ChunkChoice(
                index=0,
                finish_reason=None,
                delta=ChoiceDelta(
                    role="assistant",
                    content=content,
                    reasoning=reasoning,
                ),
            )
        ],
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
    )


def _make_empty_chunk() -> ChatCompletionChunk:
    """Create a ChatCompletionChunk with no choices."""
    return ChatCompletionChunk(
        id="test-chunk-empty",
        choices=[],
        created=1234567890,
        model="test-model",
        object="chat.completion.chunk",
    )


# --- get_chunk_content ---


def test_get_chunk_content_returns_content() -> None:
    chunk = _make_chunk(content="hello")
    assert get_chunk_content(chunk) == "hello"


def test_get_chunk_content_returns_none_when_no_content() -> None:
    chunk = _make_chunk(content=None)
    assert get_chunk_content(chunk) is None


def test_get_chunk_content_returns_none_when_no_choices() -> None:
    chunk = _make_empty_chunk()
    assert get_chunk_content(chunk) is None


# --- set_chunk_content ---


def test_set_chunk_content_sets_value() -> None:
    chunk = _make_chunk(content="old")
    result = set_chunk_content(chunk, "new")
    assert result.choices[0].delta.content == "new"


def test_set_chunk_content_sets_none() -> None:
    chunk = _make_chunk(content="old")
    result = set_chunk_content(chunk, None)
    assert result.choices[0].delta.content is None


# --- set_chunk_reasoning ---


def test_set_chunk_reasoning_sets_value() -> None:
    chunk = _make_chunk(content="answer")
    result = set_chunk_reasoning(chunk, "thinking hard")
    assert result.choices[0].delta.reasoning is not None
    assert result.choices[0].delta.reasoning.content == "thinking hard"


def test_set_chunk_reasoning_overwrites_existing() -> None:
    chunk = _make_chunk(reasoning=Reasoning(content="old reasoning"))
    result = set_chunk_reasoning(chunk, "new reasoning")
    assert result.choices[0].delta.reasoning is not None
    assert result.choices[0].delta.reasoning.content == "new reasoning"


# --- wrap_chunks_with_xml_reasoning ---


async def _collect_chunks(chunks_iter: AsyncIterator[ChatCompletionChunk]) -> list[ChatCompletionChunk]:
    """Helper to collect all chunks from an async iterator."""
    results: list[ChatCompletionChunk] = []
    async for chunk in chunks_iter:
        results.append(chunk)
    return results


async def _async_iter_chunks(chunks: list[ChatCompletionChunk]) -> AsyncIterator[ChatCompletionChunk]:
    """Create an async iterator from a list of chunks."""
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_wrap_chunks_extracts_reasoning_tags() -> None:
    """Full <think>...</think> in a single chunk is extracted as reasoning."""
    chunks = [_make_chunk(content="<think>Let me think</think>\n\nThe answer is 42.")]
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks(chunks)))

    full_content = ""
    full_reasoning = ""
    for chunk in result:
        if len(chunk.choices) > 0:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
            if chunk.choices[0].delta.reasoning:
                full_reasoning += chunk.choices[0].delta.reasoning.content

    assert full_reasoning == "Let me think"
    assert full_content.strip() == "The answer is 42."


@pytest.mark.asyncio
async def test_wrap_chunks_handles_split_tags() -> None:
    """Reasoning tags split across multiple chunks are properly handled."""
    chunks = [
        _make_chunk(content="<th"),
        _make_chunk(content="ink>"),
        _make_chunk(content="Step 1. "),
        _make_chunk(content="Step 2."),
        _make_chunk(content="</think>"),
        _make_chunk(content="\n\nFinal answer."),
    ]
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks(chunks)))

    full_content = ""
    full_reasoning = ""
    for chunk in result:
        if len(chunk.choices) > 0:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
            if chunk.choices[0].delta.reasoning:
                full_reasoning += chunk.choices[0].delta.reasoning.content

    assert full_reasoning == "Step 1. Step 2."
    assert full_content.strip() == "Final answer."


@pytest.mark.asyncio
async def test_wrap_chunks_no_reasoning_tags() -> None:
    """Chunks without reasoning tags pass through with content intact."""
    chunks = [
        _make_chunk(content="Hello "),
        _make_chunk(content="world!"),
    ]
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks(chunks)))

    full_content = ""
    for chunk in result:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
            full_content += chunk.choices[0].delta.content

    assert full_content == "Hello world!"


@pytest.mark.asyncio
async def test_wrap_chunks_passes_through_empty_choices() -> None:
    """Chunks with no choices are yielded unchanged."""
    empty = _make_empty_chunk()
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks([empty])))

    assert len(result) == 1
    assert len(result[0].choices) == 0


@pytest.mark.asyncio
async def test_wrap_chunks_passes_through_none_content() -> None:
    """Chunks with None content are yielded unchanged."""
    chunks = [_make_chunk(content=None)]
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks(chunks)))

    assert len(result) == 1
    assert result[0].choices[0].delta.content is None


@pytest.mark.asyncio
async def test_wrap_chunks_thinking_tag() -> None:
    """The <thinking> tag variant is also extracted."""
    chunks = [_make_chunk(content="<thinking>deep thought</thinking>\n\nResult.")]
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks(chunks)))

    full_content = ""
    full_reasoning = ""
    for chunk in result:
        if len(chunk.choices) > 0:
            if chunk.choices[0].delta.content:
                full_content += chunk.choices[0].delta.content
            if chunk.choices[0].delta.reasoning:
                full_reasoning += chunk.choices[0].delta.reasoning.content

    assert full_reasoning == "deep thought"
    assert full_content.strip() == "Result."
