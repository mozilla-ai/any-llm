import os

from typing_extensions import override

from any_llm.providers.openai.base import BaseOpenAIProvider


class CascadiaProvider(BaseOpenAIProvider):
    """Provider for Cascadia distributed on-prem inference clusters.

    Cascadia (https://cascadia.to) is a distributed LLM inference runtime that
    splits open-weights transformer models into INT4 OpenVINO shards and runs
    them across fleets of Intel AI PCs, keeping inference fully on-premises.
    A Cascadia coordinator exposes an OpenAI-compatible API; this provider
    targets that endpoint.
    """

    API_BASE = "http://localhost:9090/v1"
    ENV_API_KEY_NAME = "CASCADIA_API_KEY"
    ENV_API_BASE_NAME = "CASCADIA_API_BASE"
    PROVIDER_NAME = "cascadia"
    PROVIDER_DOCUMENTATION_URL = "https://cascadia.to"

    SUPPORTS_COMPLETION = True
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_MODERATION = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False
    SUPPORTS_IMAGE_GENERATION = False
    SUPPORTS_RERANK = False
    SUPPORTS_RESPONSES = False

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        # Honor an explicit key or CASCADIA_API_KEY from the environment. Cascadia
        # coordinators on trusted LANs may run keyless, so fall back to a placeholder
        # instead of hard-failing when no key is set (vLLM convention).
        return api_key or os.getenv(self.ENV_API_KEY_NAME) or "no-key-required"
