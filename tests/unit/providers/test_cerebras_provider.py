import pytest

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig


def test_stream_with_response_format_raises() -> None:
    pytest.importorskip("cerebras.cloud.sdk")
    from any_llm.providers.cerebras.cerebras import CerebrasProvider

    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    provider = CerebrasProvider(ApiConfig(api_key=api_key))

    with pytest.raises(UnsupportedParameterError):
        next(
            provider._stream_completion(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        )
