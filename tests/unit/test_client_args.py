from typing import Any
from unittest.mock import patch

import pytest

from any_llm import AnyLLM
from any_llm.constants import LLMProvider


def test_default_headers_passed_to_init_client(provider: LLMProvider) -> None:
    """Verify default_headers kwarg flows through to _init_client for all providers."""
    if provider == LLMProvider.SAGEMAKER:
        pytest.skip("sagemaker requires AWS credentials on instantiation")

    provider_class = AnyLLM.get_provider_class(provider)

    captured_kwargs: dict[str, Any] = {}

    def capture_init_client(
        _self: Any, _api_key: str | None = None, _api_base: str | None = None, **kwargs: Any
    ) -> None:
        captured_kwargs.update(kwargs)

    base_kwargs: dict[str, Any] = {
        "api_key": "test_key",
        "api_base": "https://test.example.com",
        "custom_headers": {
            "X-Custom-Header": "custom-value"
        },  # this test doesn't validate what the extra kwarg needs to be: that part is provider specific
    }

    if provider == LLMProvider.BEDROCK:
        base_kwargs["region_name"] = "us-east-1"
    if provider == LLMProvider.VERTEXAI:
        base_kwargs["project"] = "test-project"
        base_kwargs["location"] = "test-location"

    with patch.object(provider_class, "_init_client", capture_init_client):
        AnyLLM.create(provider.value, **base_kwargs)

    assert "custom_headers" in captured_kwargs, f"custom_headers not passed to {provider.value}'s _init_client"
    assert captured_kwargs["custom_headers"]["X-Custom-Header"] == "custom-value"
