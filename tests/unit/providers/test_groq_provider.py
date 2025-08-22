import pytest

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig
from any_llm.types.completion import CompletionParams


@pytest.mark.asyncio
async def test_stream_with_response_format_raises() -> None:
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    provider = GroqProvider(ApiConfig(api_key="test-api-key"))

    with pytest.raises(ValueError, match="response_format must be a pydantic model"):
        await provider.acompletion(
            CompletionParams(
                model_id="model-id",
                messages=[{"role": "user", "content": "Hello"}],
                stream=True,
                response_format={"type": "json_object"},
            )
        )


@pytest.mark.asyncio
async def test_unsupported_max_tool_calls_parameter() -> None:
    pytest.importorskip("groq")
    from any_llm.providers.groq.groq import GroqProvider

    provider = GroqProvider(ApiConfig(api_key="test-api-key"))

    with pytest.raises(UnsupportedParameterError):
        await provider.aresponses("test_model", "test_data", max_tool_calls=3)