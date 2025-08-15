from unittest.mock import Mock

import pytest

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig
from any_llm.types.completion import CompletionParams


def test_stream_with_response_format_raises() -> None:
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    provider = GroqProvider(ApiConfig(api_key="test-api-key"))

    with pytest.raises(UnsupportedParameterError):
        next(
            provider._stream_completion(
                client=Mock(),
                params=CompletionParams(
                    model_id="model-id",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=True,
                    response_format={"type": "json_object"},
                ),
            )
        )


def test_unsupported_max_tool_calls_parameter() -> None:
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    provider = GroqProvider(ApiConfig(api_key="test-api-key"))

    with pytest.raises(UnsupportedParameterError):
        provider.responses("test_model", "test_data", max_tool_calls=3)
