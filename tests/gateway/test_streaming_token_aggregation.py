"""Tests for streaming token aggregation logic."""

from any_llm.types.completion import CompletionUsage


def test_last_nonzero_value_aggregation_cumulative() -> None:
    """Test that last-non-zero-value works for cumulative reporting providers.

    Cumulative providers send increasing totals: [10, 20, 30].
    Taking the last non-zero value gives 30 (correct total).
    """
    chunks_usage = [
        CompletionUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
        CompletionUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        CompletionUsage(prompt_tokens=100, completion_tokens=30, total_tokens=130),
    ]

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    for usage in chunks_usage:
        if usage.prompt_tokens:
            prompt_tokens = usage.prompt_tokens
        if usage.completion_tokens:
            completion_tokens = usage.completion_tokens
        if usage.total_tokens:
            total_tokens = usage.total_tokens

    assert prompt_tokens == 100
    assert completion_tokens == 30  # Last value (cumulative total)
    assert total_tokens == 130


def test_last_nonzero_value_aggregation_final_chunk_only() -> None:
    """Test that last-non-zero-value works for final-chunk-only providers.

    Providers that only report on the final chunk send usage once.
    Taking the last non-zero value gives that single value (correct).
    """
    # Most chunks have no usage, final chunk has the totals
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    # Simulate: only the last chunk has usage
    final_usage = CompletionUsage(prompt_tokens=50, completion_tokens=200, total_tokens=250)
    if final_usage.prompt_tokens:
        prompt_tokens = final_usage.prompt_tokens
    if final_usage.completion_tokens:
        completion_tokens = final_usage.completion_tokens
    if final_usage.total_tokens:
        total_tokens = final_usage.total_tokens

    assert prompt_tokens == 50
    assert completion_tokens == 200
    assert total_tokens == 250
