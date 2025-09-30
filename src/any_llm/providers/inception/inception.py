from typing import Any

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import CompletionParams


class InceptionProvider(BaseOpenAIProvider):
    API_BASE = "https://api.inceptionlabs.ai/v1"
    ENV_API_KEY_NAME = "INCEPTION_API_KEY"
    PROVIDER_NAME = "inception"
    PROVIDER_DOCUMENTATION_URL = "https://inceptionlabs.ai/"

    SUPPORTS_EMBEDDING = False  # Inception doesn't host an embedding model
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        if params.response_format is not None:
            param = "response_format"
            raise UnsupportedParameterError(param, "inception")
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages"})
        converted_params.update(kwargs)
        return converted_params
