import os

from any_llm.provider import ProviderName

LOCAL_PROVIDERS = [ProviderName.LLAMACPP, ProviderName.OLLAMA, ProviderName.LMSTUDIO, ProviderName.LLAMAFILE]

EXPECTED_PROVIDERS = os.environ.get("EXPECTED_PROVIDERS", "").split(",")
