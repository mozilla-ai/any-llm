from typing import Any

from typing_extensions import override

from any_llm.logging import logger
from any_llm.types.completion import CompletionParams

from .base import BaseOpenAIProvider


class OpenaiProvider(BaseOpenAIProvider):
    API_BASE = "https://api.openai.com/v1"

    ENV_API_KEY_NAME = "OPENAI_API_KEY"
    ENV_API_BASE_NAME = "OPENAI_BASE_URL"
    PROVIDER_NAME = "openai"
    PROVIDER_DOCUMENTATION_URL = "https://platform.openai.com/docs/api-reference"
    SUPPORTS_RESPONSES = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = True

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        converted_params = BaseOpenAIProvider._convert_completion_params(params, **kwargs)
        if "max_tokens" in converted_params:
            max_tokens = converted_params.pop("max_tokens")
            if "max_completion_tokens" in converted_params:
                logger.warning(
                    "Ignoring max_tokens (%s) in favor of max_completion_tokens (%s).",
                    max_tokens,
                    converted_params["max_completion_tokens"],
                )
            else:
                converted_params["max_completion_tokens"] = max_tokens
        return converted_params
