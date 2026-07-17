from collections.abc import Sequence
from typing import Any

from typing_extensions import override

from any_llm.providers.edenai.utils import _convert_models_list
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.model import Model


class EdenaiProvider(BaseOpenAIProvider):
    API_BASE = "https://api.edenai.run/v3"
    ENV_API_KEY_NAME = "EDENAI_API_KEY"
    ENV_API_BASE_NAME = "EDENAI_API_BASE"
    PROVIDER_NAME = "edenai"
    PROVIDER_DOCUMENTATION_URL = "https://www.edenai.co/docs"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = False
    # Eden AI's OpenAI-compatible endpoint does not surface structured reasoning
    # content in a separate field (it is inlined in the message content), so
    # reasoning is reported as unsupported here.
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_EMBEDDING = True
    SUPPORTS_MODERATION = True

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Convert the Eden AI /v3/models response to valid Model objects."""
        return _convert_models_list(response)
