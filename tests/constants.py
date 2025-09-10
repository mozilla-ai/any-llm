import os

from any_llm.constants import ProviderName

LOCAL_PROVIDERS = [ProviderName.LLAMACPP, ProviderName.OLLAMA, ProviderName.LMSTUDIO, ProviderName.LLAMAFILE]

EXPECTED_PROVIDERS = os.environ.get("EXPECTED_PROVIDERS", "").split(",")

INCLUDE_LOCAL_PROVIDERS = os.getenv("INCLUDE_LOCAL_PROVIDERS", "true").lower() in ("true", "1", "t")

INCLUDE_NON_LOCAL_PROVIDERS = os.getenv("INCLUDE_NON_LOCAL_PROVIDERS", "true").lower() in ("true", "1", "t")
