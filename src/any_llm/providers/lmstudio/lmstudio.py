from any_llm.providers.openai.base import BaseOpenAIProvider

# LM Studio has a python sdk, but per their docs they are compliant with OpenAI spec
# https://lmstudio.ai/docs/app/api/endpoints/openai
# So until its clear why the python sdk should be used, we'll default to inheriting from OpenAI SDK.


class LmstudioProvider(BaseOpenAIProvider):
    API_BASE = "http://localhost:1234/v1"
    ENV_API_KEY_NAME = "LM_STUDIO_API_KEY"
    PROVIDER_NAME = "lmstudio"
    PROVIDER_DOCUMENTATION_URL = "https://lmstudio.ai/"

    SUPPORTS_COMPLETION_REASONING = True

    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        return ""
