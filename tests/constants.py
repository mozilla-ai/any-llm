import os

from any_llm.constants import LLMProvider

LOCAL_PROVIDERS = [
    LLMProvider.LLAMACPP,
    LLMProvider.OLLAMA,
    LLMProvider.LMSTUDIO,
    LLMProvider.LLAMAFILE,
    LLMProvider.CASCADIA,
]

# Providers that should never run in CI (only for local development)
CI_EXCLUDED_PROVIDERS = [
    LLMProvider.AZUREANTHROPIC,
    LLMProvider.VERTEXAIANTHROPIC,
    LLMProvider.VLLM,
    LLMProvider.CASCADIA,
]

# Strip whitespace and drop empties so values like "anthropic, otari" or an unset env var
# (which would otherwise yield [""]) compare cleanly in `provider in EXPECTED_PROVIDERS` checks.
EXPECTED_PROVIDERS = [
    provider.strip() for provider in os.environ.get("EXPECTED_PROVIDERS", "").split(",") if provider.strip()
]

INCLUDE_LOCAL_PROVIDERS = os.getenv("INCLUDE_LOCAL_PROVIDERS", "true").lower() in ("true", "1", "t")

INCLUDE_NON_LOCAL_PROVIDERS = os.getenv("INCLUDE_NON_LOCAL_PROVIDERS", "true").lower() in ("true", "1", "t")
