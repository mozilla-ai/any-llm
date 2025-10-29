from any_llm.providers.openai.base import BaseOpenAIProvider


class MinimaxProvider(BaseOpenAIProvider):
    API_BASE = "https://api.minimax.io/v1"
    ENV_API_KEY_NAME = "MINIMAX_API_KEY"
    PROVIDER_NAME = "minimax"
    PROVIDER_DOCUMENTATION_URL = "https://www.minimax.io/platform_overview"

    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_COMPLETION_REASONING = True
