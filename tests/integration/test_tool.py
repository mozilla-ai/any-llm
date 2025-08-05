from typing import Any
import httpx
import pytest
from any_llm import completion, ProviderName
from any_llm.exceptions import MissingApiKeyError
from openai import APIConnectionError


def test_tool(
    provider: ProviderName,
    provider_model_map: dict[ProviderName, str],
    provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]],
) -> None:
    """Test that all supported providers can be loaded successfully."""
    model_id = provider_model_map[provider]
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})

    def capital_city(country: str) -> str:
        """Tool function to get the capital of a city."""
        return f"The capital of {country} is what you want it to be"

    prompt = "Please call the `capital_city` tool with the argument `France`"
    try:
        result = completion(
            f"{provider.value}/{model_id}",
            **extra_kwargs,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            tools=[capital_city],
        )
        assert any(choice.message.tool_calls is not None for choice in result.choices)  # type: ignore[union-attr]
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in [ProviderName.OLLAMA, ProviderName.LMSTUDIO]:
            pytest.skip("Local Model host is not set up, skipping")
        raise
