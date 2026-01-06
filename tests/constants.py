import os

from any_llm.constants import LLMProvider

LOCAL_PROVIDERS = [
    LLMProvider.LLAMACPP,
    LLMProvider.OLLAMA,
    LLMProvider.LMSTUDIO,
    LLMProvider.LLAMAFILE,
    LLMProvider.GATEWAY,
]

# Providers that should never run in CI (only for local development)
CI_EXCLUDED_PROVIDERS = [
    LLMProvider.VLLM,
]

EXPECTED_PROVIDERS = os.environ.get("EXPECTED_PROVIDERS", "").split(",")

INCLUDE_LOCAL_PROVIDERS = os.getenv("INCLUDE_LOCAL_PROVIDERS", "true").lower() in ("true", "1", "t")

INCLUDE_NON_LOCAL_PROVIDERS = os.getenv("INCLUDE_NON_LOCAL_PROVIDERS", "true").lower() in ("true", "1", "t")
