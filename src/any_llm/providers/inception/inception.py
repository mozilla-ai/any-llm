import warnings
from typing import Any

from typing_extensions import override

from any_llm.exceptions import UnsupportedParameterError
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import CompletionParams


class InceptionProvider(BaseOpenAIProvider):
    API_BASE = "https://api.inceptionlabs.ai/v1"
    ENV_API_KEY_NAME = "INCEPTION_API_KEY"
    ENV_API_BASE_NAME = "INCEPTION_API_BASE"
    PROVIDER_NAME = "inception"
    PROVIDER_DOCUMENTATION_URL = "https://inceptionlabs.ai/"

    SUPPORTS_EMBEDDING = False  # Inception doesn't host an embedding model
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_COMPLETION_RESPONSE_FORMAT = False

    @override
    def __init__(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        # FutureWarning because AnyLLM.create builds the provider, and default filters hide a
        # DeprecationWarning raised from inside the library.
        warnings.warn(
            "Inception stops serving the Mercury API after December 28, 2026 "
            "(https://docs.inceptionlabs.ai/support/wind-down), so the inception provider "
            "stops working after that date.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(api_key=api_key, api_base=api_base, **kwargs)

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        if params.response_format is not None:
            param = "response_format"
            raise UnsupportedParameterError(param, "inception")
        return BaseOpenAIProvider._convert_completion_params(params, **kwargs)
