from any_llm.utils.usage import completion_usage_from_dict


def test_keeps_nested_cached_and_reasoning_tokens() -> None:
    usage = completion_usage_from_dict(
        {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_tokens_details": {"cached_tokens": 64},
            "completion_tokens_details": {"reasoning_tokens": 4},
        }
    )

    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 110
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 64
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 4


def test_falls_back_to_top_level_cached_tokens() -> None:
    usage = completion_usage_from_dict({"prompt_tokens": 100, "cached_tokens": 64})

    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 64


def test_falls_back_to_top_level_cached_tokens_when_nested_is_null() -> None:
    usage = completion_usage_from_dict(
        {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": None}, "cached_tokens": 64}
    )

    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 64


def test_null_counts_become_zero_and_details_stay_unset() -> None:
    usage = completion_usage_from_dict(
        {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        }
    )

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0
    assert usage.prompt_tokens_details is None
    assert usage.completion_tokens_details is None
