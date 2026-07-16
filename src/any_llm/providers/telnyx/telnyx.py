from any_llm.providers.openai.base import BaseOpenAIProvider


class TelnyxProvider(BaseOpenAIProvider):
    API_BASE = "https://api.telnyx.com/v2/ai"
    ENV_API_KEY_NAME = "TELNYX_API_KEY"
    ENV_API_BASE_NAME = "TELNYX_API_BASE"
    PROVIDER_NAME = "telnyx"
    PROVIDER_DOCUMENTATION_URL = "https://developers.telnyx.com/docs/inference/getting-started"

    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_PDF = False
