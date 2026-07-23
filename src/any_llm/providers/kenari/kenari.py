from any_llm.providers.openai.base import BaseOpenAIProvider


class KenariProvider(BaseOpenAIProvider):
    """Provider for Kenari (https://kenari.id).

    Kenari is an OpenAI-compatible LLM gateway focused on the Indonesian market,
    serving Claude, GPT, Gemini, DeepSeek, Kimi, GLM, Qwen and other models behind
    one API with local-currency billing. It exposes the standard chat completions
    and model listing endpoints under https://kenari.id/v1, so this provider only
    needs to set configuration defaults and capability flags on top of
    BaseOpenAIProvider.
    """

    API_BASE = "https://kenari.id/v1"
    ENV_API_KEY_NAME = "KENARI_API_KEY"
    ENV_API_BASE_NAME = "KENARI_API_BASE"
    PROVIDER_NAME = "kenari"
    PROVIDER_DOCUMENTATION_URL = "https://kenari.id/docs"

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
