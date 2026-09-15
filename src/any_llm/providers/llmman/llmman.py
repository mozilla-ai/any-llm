from typing_extensions import override

from any_llm.providers.openai.base import BaseOpenAIProvider


class LlmmanProvider(BaseOpenAIProvider):
    """Provider for llmman, a local model runner (https://github.com/llmmanorg/llmman).

    llmman serves the Ollama API alongside OpenAI- and Anthropic-compatible ones on
    port 17434, backed by upstream llama.cpp (llama-server), vllm or mlx-lm. This
    provider targets the OpenAI-compatible ``/v1`` routes, so no extra SDK is needed.
    """

    API_BASE = "http://localhost:17434/v1"
    ENV_API_KEY_NAME = "None"
    ENV_API_BASE_NAME = "LLMMAN_API_BASE"
    PROVIDER_NAME = "llmman"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/llmmanorg/llmman"

    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_MODERATION = False

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        # llmman runs locally and does not require an API key.
        return "no-key-required"
