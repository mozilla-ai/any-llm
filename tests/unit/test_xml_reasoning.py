from collections.abc import AsyncIterator
from typing import Any, Literal

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
from any_llm.utils.reasoning import partial_reasoning_tag_suffix_len


def _make_chunk(
    content: str | None = None,
    reasoning: Reasoning | None = None,
    finish_reason: Literal["stop", "length"] | None = None,
    role: Literal["assistant"] | None = "assistant",
    extra_content: dict[str, Any] | None = None,
) -> ChatCompletionChunk:
    """Create a minimal ChatCompletionChunk for testing."""
    return ChatCompletionChunk(
        id="test-chunk",
        choices=[
            ChunkChoice(
                index=0,
                finish_reason=finish_reason,
                delta=ChoiceDelta(
                    role=role,
                    content=content,
                    reasoning=reasoning,
                    extra_content=extra_content,
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


def _accumulate(chunks: list[ChatCompletionChunk]) -> tuple[str, str]:
    """Join the content and reasoning deltas of processed chunks."""
    content = ""
    reasoning = ""
    for chunk in chunks:
        if not chunk.choices:
            continue
        content += chunk.choices[0].delta.content or ""
        if chunk.choices[0].delta.reasoning:
            reasoning += chunk.choices[0].delta.reasoning.content
    return content, reasoning


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parts", "expected_content", "expected_reasoning"),
    [
        (["<think>reasoning</th", "ink>answer"], "answer", "reasoning"),
        (["preface <th", "ink>reasoning</think>answer"], "preface answer", "reasoning"),
        (["preface <thin", "king>reasoning</thinking>answer"], "preface answer", "reasoning"),
        (["a<b <think>reasoning</thin", "k>answer"], "a<b answer", "reasoning"),
        (["<think>a</think>mid <th", "ink>b</think>end"], "mid end", "ab"),
    ],
)
async def test_wrap_chunks_handles_tags_split_after_content(
    parts: list[str],
    expected_content: str,
    expected_reasoning: str,
) -> None:
    """Tags split across chunks are handled even when preceded by other text."""
    chunks = [_make_chunk(content=part) for part in parts]
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks(chunks)))

    assert _accumulate(result) == (expected_content, expected_reasoning)


@pytest.mark.asyncio
async def test_wrap_chunks_flushes_trailing_partial_opening_tag() -> None:
    """A stream that ends on a partial opening tag still emits the held-back text."""
    chunks = [_make_chunk(content="trailing partial <th")]
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks(chunks)))

    assert _accumulate(result) == ("trailing partial <th", "")


@pytest.mark.asyncio
async def test_wrap_chunks_preserves_metadata_for_pure_partial_opening_tag() -> None:
    """A source chunk held entirely for EOF flushing keeps its metadata."""
    chunks = [_make_chunk(content="<th", extra_content={"source": "partial"})]
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks(chunks)))

    assert _accumulate(result) == ("<th", "")
    assert len(result) == 1
    assert result[0].choices[0].delta.role == "assistant"
    assert result[0].choices[0].delta.extra_content == {"source": "partial"}


@pytest.mark.asyncio
async def test_wrap_chunks_flushes_unterminated_reasoning() -> None:
    """A reasoning block with no closing tag is emitted as reasoning, not dropped."""
    chunks = [_make_chunk(content="<think>unterminated "), _make_chunk(content="reasoning")]
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks(chunks)))

    assert _accumulate(result) == ("", "unterminated reasoning")


@pytest.mark.asyncio
async def test_wrap_chunks_flushes_partial_closing_tag() -> None:
    """A partial closing tag is retained as reasoning when the stream ends."""
    chunks = [_make_chunk(content="<think>reasoning</th")]
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks(chunks)))

    assert _accumulate(result) == ("", "reasoning</th")


@pytest.mark.asyncio
async def test_wrap_chunks_flushes_before_contentless_terminal_chunk() -> None:
    """Buffered content is emitted before a terminal chunk without content."""
    chunks = [
        _make_chunk(content="prefix <th", extra_content={"source": "content"}),
        _make_chunk(finish_reason="stop", role=None),
    ]
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks(chunks)))

    assert _accumulate(result) == ("prefix <th", "")
    assert result[-1].choices[0].finish_reason == "stop"
    assert [chunk.choices[0].delta.extra_content for chunk in result if chunk.choices[0].delta.extra_content] == [
        {"source": "content"}
    ]


@pytest.mark.asyncio
async def test_wrap_chunks_flushes_once_when_content_chunk_is_terminal() -> None:
    """A terminal content chunk keeps its metadata while its reasoning is flushed."""
    chunks = [_make_chunk(content="<think>reasoning", finish_reason="length")]
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks(chunks)))

    assert _accumulate(result) == ("", "reasoning")
    assert len(result) == 1
    assert result[0].choices[0].finish_reason == "length"


@pytest.mark.asyncio
async def test_wrap_chunks_preserves_terminal_metadata_after_partial_tag() -> None:
    """A terminal chunk completing an earlier partial tag keeps its finish reason."""
    chunks = [
        _make_chunk(content="prefix <th"),
        _make_chunk(content="ink>reasoning</think>answer", finish_reason="stop", role=None),
    ]
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks(chunks)))

    assert _accumulate(result) == ("prefix answer", "reasoning")
    assert [chunk.choices[0].finish_reason for chunk in result] == [None, "stop"]


@pytest.mark.asyncio
async def test_wrap_chunks_empty_stream_yields_nothing() -> None:
    """No chunks in means no chunks out, including no flush chunk."""
    result = await _collect_chunks(wrap_chunks_with_xml_reasoning(_async_iter_chunks([])))

    assert result == []


@pytest.mark.parametrize(
    ("text", "tag_kind", "expected"),
    [
        ("<th", "opening", 3),
        ("preface <th", "opening", 3),
        ("reasoning</th", "closing", 4),
        ("<think>", "opening", 0),
        ("plain text", "opening", 0),
        ("", "opening", 0),
        ("a<b<thin", "opening", 5),
    ],
)
def test_partial_reasoning_tag_suffix_len(
    text: str,
    tag_kind: Literal["opening", "closing"],
    expected: int,
) -> None:
    assert partial_reasoning_tag_suffix_len(text, tag_kind=tag_kind) == expected
