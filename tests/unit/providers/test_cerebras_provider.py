import sys
from unittest.mock import patch

import pytest

from any_llm.exceptions import UnsupportedParameterError
from any_llm.provider import ApiConfig, ProviderFactory
from any_llm.providers.cerebras.cerebras import CerebrasProvider


@pytest.mark.asyncio
async def test_stream_with_response_format_raises() -> None:
    api_key = "test-api-key"
    model = "model-id"
    messages = [{"role": "user", "content": "Hello"}]

    provider = CerebrasProvider(ApiConfig(api_key=api_key))

    chunks = provider._stream_completion_async(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )
    with pytest.raises(UnsupportedParameterError):
        async for _ in chunks:
            pass


def test_provider_with_no_packages_installed() -> None:
    with patch.dict(sys.modules, dict.fromkeys(["cerebras"])):
        for mod in list(sys.modules):
            if mod.startswith("any_llm.providers.cerebras"):
                sys.modules.pop(mod)
        try:
            import any_llm.providers.cerebras  # noqa: F401
        except ImportError:
            pytest.fail("Import raised an unexpected ImportError")


def test_call_to_provider_with_no_packages_installed() -> None:
    packages = ["cerebras", "instructor"]
    with patch.dict(sys.modules, dict.fromkeys(packages)):
        for mod in list(sys.modules):
            if mod.startswith("any_llm.providers.cerebras"):
                sys.modules.pop(mod)
        with pytest.raises(ImportError, match="cerebras required packages are not installed"):
            ProviderFactory.create_provider("cerebras", ApiConfig())
