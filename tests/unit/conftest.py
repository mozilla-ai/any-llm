from functools import cache

import pytest

from any_llm import AnyLLM

# Read directly by providers or their SDKs rather than through ENV_API_KEY_NAME / ENV_API_BASE_NAME.
_EXTRA_ENV_VARS = (
    "ANY_LLM_UNIFIED_EXCEPTIONS",
    "AZURE_ANTHROPIC_RESOURCE",
    "GOOGLE_GENAI_USE_VERTEXAI",
)


@cache
def _provider_env_vars() -> frozenset[str]:
    names: set[str] = set(_EXTRA_ENV_VARS)
    for provider_key in AnyLLM.get_supported_providers():
        try:
            provider_class = AnyLLM.get_provider_class(provider_key)
        except ImportError:
            continue
        names.update((provider_class.ENV_API_KEY_NAME, provider_class.ENV_API_BASE_NAME or ""))
    # Some providers use "" or a human-readable description here instead of a variable name.
    return frozenset(name for name in names if name.isidentifier())


@pytest.fixture(autouse=True)
def clear_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent of provider credentials and endpoints exported in the developer's shell."""
    for name in _provider_env_vars():
        monkeypatch.delenv(name, raising=False)
