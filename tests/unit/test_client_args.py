import sys
from typing import Any
from unittest.mock import patch

import pytest

from any_llm import AnyLLM
from any_llm.constants import LLMProvider


def test_default_headers_passed_to_init_client(provider: LLMProvider) -> None:
    """Verify provider kwargs reach _init_client, and that library options do not.

    ``custom_headers`` stands in for any provider-specific kwarg, which must arrive at
    ``_init_client`` untouched. ``unified_exceptions`` is the opposite case: it configures
    any-llm rather than the SDK, so it must land on the instance and never be forwarded,
    or every provider's client constructor would reject it.
    """
    if provider == LLMProvider.SAGEMAKER:
        pytest.skip("sagemaker requires AWS credentials on instantiation")
    if sys.version_info >= (3, 14) and provider.value in ("voyage", "watsonx"):
        pytest.skip(f"{provider.value} is not compatible with Python 3.14+")

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
        "unified_exceptions": True,
    }

    if provider == LLMProvider.BEDROCK:
        base_kwargs["region_name"] = "us-east-1"
    if provider == LLMProvider.VERTEXAI:
        base_kwargs["project"] = "test-project"
        base_kwargs["location"] = "test-location"

    with patch.object(provider_class, "_init_client", capture_init_client):
        instance = AnyLLM.create(provider.value, **base_kwargs)

    assert "custom_headers" in captured_kwargs, f"custom_headers not passed to {provider.value}'s _init_client"
    assert captured_kwargs["custom_headers"]["X-Custom-Header"] == "custom-value"

    assert "unified_exceptions" not in captured_kwargs, (
        f"unified_exceptions leaked into {provider.value}'s _init_client"
    )
    assert instance._unified_exceptions is True, (
        f"{provider.value} dropped unified_exceptions instead of storing it on the instance"
    )
