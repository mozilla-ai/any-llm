from typing import Any

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import CompletionParams


class ZaiProvider(BaseOpenAIProvider):
    """
    Provider for z.ai API.

    z.ai is an OpenAI-compatible API that provides access to various AI models.
    """

    PROVIDER_NAME = "zai"
    API_BASE = "https://api.z.ai/api/paas/v4/"
    PROVIDER_DOCUMENTATION_URL = "https://docs.z.ai/guides/develop/python/introduction"
    ENV_API_KEY_NAME = "ZAI_API_KEY"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True

    PACKAGES_INSTALLED = True

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        # response_format is supported in the z.ai SDK, but the SDK doesn't yet have an async client
        # so we can't use it in any-llm
        if params.response_format is not None:
            param = "response_format"
            raise UnsupportedParameterError(param, "zai")
        # Copy of the logic from the base implementation because you can't use super() in a static method
        converted_params = params.model_dump(exclude_none=True, exclude={"model_id", "messages"})
        converted_params.update(kwargs)
        return converted_params
