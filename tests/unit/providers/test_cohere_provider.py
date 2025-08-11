from typing import Any

import pytest

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig


def _mk_provider() -> Any:
    pytest.importorskip("cohere")
    from any_llm.providers.cohere.cohere import CohereProvider

    return CohereProvider(ApiConfig(api_key="test-api-key"))


def test_response_format_raises_for_non_streaming() -> None:
    provider = _mk_provider()

    with pytest.raises(UnsupportedParameterError):
        provider.completion(
            model="model-id",
            messages=[{"role": "user", "content": "Hello"}],
            response_format={"type": "json_object"},
        )


def test_stream_and_response_format_combination_raises() -> None:
    provider = _mk_provider()

    with pytest.raises(UnsupportedParameterError):
        provider.completion(
            model="model-id",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
            response_format={"type": "json_object"},
        )


def test_parallel_tool_calls_raises() -> None:
    provider = _mk_provider()

    with pytest.raises(UnsupportedParameterError):
        provider.completion(
            model="model-id",
            messages=[{"role": "user", "content": "Hello"}],
            parallel_tool_calls=False,
        )
