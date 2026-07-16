from typing import Any

from typing_extensions import override

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import CompletionParams


class GmiProvider(BaseOpenAIProvider):
    API_BASE = "https://api.gmi-serving.com/v1"
    ENV_API_KEY_NAME = "GMI_API_KEY"
    ENV_API_BASE_NAME = "GMI_API_BASE"
    PROVIDER_NAME = "gmi"
    PROVIDER_DOCUMENTATION_URL = "https://docs.gmicloud.ai/inference-engine/api-reference/llm-api-reference"

    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_MODERATION = False
    SUPPORTS_RESPONSES = False
    SUPPORTS_BATCH = False
    SUPPORTS_IMAGE_GENERATION = False
    SUPPORTS_RERANK = False

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        converted_params = BaseOpenAIProvider._convert_completion_params(params, **kwargs)
        if "max_completion_tokens" in converted_params:
            converted_params["max_tokens"] = converted_params.pop("max_completion_tokens")
        return converted_params
