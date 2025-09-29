from typing import Any

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.providers.openrouter.utils import build_reasoning_directive
from any_llm.types.completion import CompletionParams


class OpenrouterProvider(BaseOpenAIProvider):
    API_BASE = "https://openrouter.ai/api/v1"
    ENV_API_KEY_NAME = "OPENROUTER_API_KEY"
    PROVIDER_NAME = "openrouter"
    PROVIDER_DOCUMENTATION_URL = "https://openrouter.ai/docs"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_EMBEDDING = False

    @staticmethod
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """Convert CompletionParams to kwargs for OpenRouter API, including reasoning directive."""
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
