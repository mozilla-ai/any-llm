from any_llm.providers.openai.base import BaseOpenAIProvider


class AzureopenaiProvider(BaseOpenAIProvider):
    """Azure OpenAI AnyLLM."""

    ENV_API_KEY_NAME = "AZURE_OPENAI_API_KEY"
    PROVIDER_NAME = "azureopenai"
    PROVIDER_DOCUMENTATION_URL = "https://learn.microsoft.com/en-us/azure/ai-foundry/"
    SUPPORTS_RESPONSES = True
    SUPPORTS_LIST_MODELS = True

    def _init_client(self) -> None:
        if not self.config.client_args:
            self.config.client_args = {}
        self.config.client_args["default_query"] = {"api-version": "preview"}
        return super()._init_client()
