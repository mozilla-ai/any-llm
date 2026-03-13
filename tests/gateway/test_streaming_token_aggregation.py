"""Tests for streaming token aggregation logic."""

from any_llm.gateway.streaming import _merge_usage
from any_llm.types.completion import CompletionUsage


def test_merge_usage_cumulative() -> None:
    """Test that _merge_usage keeps last non-zero value for cumulative providers."""
    chunks = [
        CompletionUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        CompletionUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        CompletionUsage(prompt_tokens=100, completion_tokens=30, total_tokens=130),
    ]

    result = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    for chunk in chunks:
        result = _merge_usage(result, chunk)

    assert result.prompt_tokens == 100
    assert result.completion_tokens == 30
    assert result.total_tokens == 130


def test_merge_usage_final_chunk_only() -> None:
    """Test that _merge_usage works when only the last chunk has usage."""
    base = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    final = CompletionUsage(prompt_tokens=50, completion_tokens=200, total_tokens=250)

    result = _merge_usage(base, final)

    assert result.prompt_tokens == 50
    assert result.completion_tokens == 200
    assert result.total_tokens == 250


def test_merge_usage_preserves_current_on_zero_update() -> None:
    """Test that _merge_usage preserves current values when update has zeros."""
    current = CompletionUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    update = CompletionUsage(prompt_tokens=0, completion_tokens=75, total_tokens=0)

    result = _merge_usage(current, update)

    assert result.prompt_tokens == 100
    assert result.completion_tokens == 75
    assert result.total_tokens == 150
