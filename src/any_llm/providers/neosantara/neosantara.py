from any_llm.providers.openai.base import BaseOpenAIProvider


class NeosantaraProvider(BaseOpenAIProvider):
    API_BASE = "https://api.neosantara.xyz/v1"
    ENV_API_KEY_NAME = "NEOSANTARA_API_KEY"
    ENV_API_BASE_NAME = "NEOSANTARA_API_BASE"
    PROVIDER_NAME = "neosantara"
    PROVIDER_DOCUMENTATION_URL = "https://docs.neosantara.xyz"

    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_IMAGE_GENERATION = True
    SUPPORTS_AUDIO_TRANSCRIPTION = True
    SUPPORTS_AUDIO_SPEECH = True
    SUPPORTS_BATCH = True
