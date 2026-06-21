from collections.abc import Sequence
from typing import Any

from typing_extensions import override

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.requesty.utils import _convert_models_list, build_reasoning_directive
from any_llm.types.completion import CompletionParams
from any_llm.types.model import Model


class RequestyProvider(BaseOpenAIProvider):
    API_BASE = "https://router.requesty.ai/v1"
    ENV_API_KEY_NAME = "REQUESTY_API_KEY"
    ENV_API_BASE_NAME = "REQUESTY_API_BASE"
    PROVIDER_NAME = "requesty"
    PROVIDER_DOCUMENTATION_URL = "https://docs.requesty.ai"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_EMBEDDING = True
    # Requesty does not expose a moderation endpoint.
    SUPPORTS_MODERATION = False

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert Requesty list models response to valid Model objects."""
        return _convert_models_list(response)

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for Requesty API, including reasoning directive."""
        converted_params = BaseOpenAIProvider._convert_completion_params(params, **kwargs)

        reasoning_directive = build_reasoning_directive(
            reasoning=kwargs.get("reasoning"),
            reasoning_effort=params.reasoning_effort,
        )

        if reasoning_directive is not None:
            converted_params.pop("reasoning_effort", None)
            extra_body = converted_params.get("extra_body", {}).copy()
            extra_body["reasoning"] = reasoning_directive
            converted_params["extra_body"] = extra_body

        return converted_params
