from any_llm.providers.openai.base import BaseOpenAIProvider


class AtlascloudProvider(BaseOpenAIProvider):
    """Provider for Atlas Cloud (https://www.atlascloud.ai).

    Atlas Cloud is an OpenAI-compatible inference platform serving DeepSeek, Qwen,
    Kimi, GLM and MiniMax models. It exposes the standard chat completions and model
    listing endpoints under https://api.atlascloud.ai/v1, so this provider only needs
    to set configuration defaults and capability flags on top of BaseOpenAIProvider.
    """

    API_BASE = "https://api.atlascloud.ai/v1"
    ENV_API_KEY_NAME = "ATLASCLOUD_API_KEY"
    ENV_API_BASE_NAME = "ATLASCLOUD_API_BASE"
    PROVIDER_NAME = "atlascloud"
    PROVIDER_DOCUMENTATION_URL = "https://www.atlascloud.ai/docs"

    SUPPORTS_COMPLETION = True
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_MODERATION = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False
    SUPPORTS_IMAGE_GENERATION = False
    SUPPORTS_RERANK = False
    SUPPORTS_RESPONSES = False
