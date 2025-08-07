from typing import Any
import httpx
import pytest
from any_llm import completion, ProviderName
from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.types.completion import ChatCompletionChunk
from openai import APIConnectionError


def test_streaming_completion(
    provider: ProviderName,
    provider_model_map: dict[ProviderName, str],
    provider_extra_kwargs_map: dict[ProviderName, dict[str, Any]],
) -> None:
    """Test that streaming completion works for supported providers."""
    model_id = provider_model_map[provider]
    extra_kwargs = provider_extra_kwargs_map.get(provider, {})
    try:
        output = ""
        reasoning = ""
        num_chunks = 0
        for result in completion(
            f"{provider.value}/{model_id}",
            **extra_kwargs,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that exactly follows the user request."},
                {"role": "user", "content": "Say the exact phrase:'Hello World'"},
            ],
            stream=True,
            temperature=0.1,
        ):
            num_chunks += 1
            # Verify the response is still a valid ChatCompletion object
            assert isinstance(result, ChatCompletionChunk)
            if len(result.choices) > 0:
                output += result.choices[0].delta.content or ""
                if result.choices[0].delta.reasoning:
                    reasoning += result.choices[0].delta.reasoning.content or ""
        assert num_chunks >= 2, f"Expected at least 2 chunks, got {num_chunks}"
        assert "hello world" in output.lower()
    except MissingApiKeyError:
        pytest.skip(f"{provider.value} API key not provided, skipping")
    except UnsupportedParameterError:
        pytest.skip(f"Streaming is not supported for {provider.value}")
    except (httpx.HTTPStatusError, httpx.ConnectError, APIConnectionError):
        if provider in [ProviderName.OLLAMA, ProviderName.LMSTUDIO]:
            pytest.skip("Local Model host is not set up, skipping")
        raise
