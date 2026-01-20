from enum import Enum


class Providers(str, Enum):
    """Enum of supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    MISTRAL = "mistral"
    GEMINI = "gemini"
    VERTEXAI = "vertexai"
    COHERE = "cohere"
    CEREBRAS = "cerebras"
    GROQ = "groq"
    BEDROCK = "bedrock"
    AZURE = "azure"
    AZURE_OPENAI = "azureopenai"
    WATSONX = "watsonx"
    TOGETHER = "together"
    SAMBANOVA = "sambanova"
    OLLAMA = "ollama"
    MOONSHOT = "moonshot"
    NEBIUS = "nebius"
    XAI = "xai"
    DATABRICKS = "databricks"
    DEEPSEEK = "deepseek"
    INCEPTION = "inception"
    OPENROUTER = "openrouter"
    PORTKEY = "portkey"
    LMSTUDIO = "lmstudio"
    LLAMA = "llama"
    VOYAGE = "voyage"
    PERPLEXITY = "perplexity"
    FIREWORKS = "fireworks"
    HUGGINGFACE = "huggingface"
    LLAMAFILE = "llamafile"
    LLAMACPP = "llamacpp"
    SAGEMAKER = "sagemaker"
    MINIMAX = "minimax"
    VLLM = "vllm"
    ZAI = "zai"


# Mapping of provider enum to module name and class name
PROVIDER_REGISTRY: dict[str, tuple[str, str]] = {
    Providers.OPENAI.value: ("any_llm.providers.openai", "OpenAIProvider"),
}


def get_provider_class(provider: str | Providers) -> type:
    """Dynamically load and return the provider class.

    Args:
        provider: The provider name or enum value.

    Returns:
        The provider class.

    Raises:
        ValueError: If the provider is not supported.

    """
    import importlib

    provider_value = provider.value if isinstance(provider, Providers) else provider

    if provider_value not in PROVIDER_REGISTRY:
        msg = f"Provider '{provider_value}' is not supported"
        raise ValueError(msg)

    module_name, class_name = PROVIDER_REGISTRY[provider_value]
    module = importlib.import_module(module_name)
    provider_class: type = getattr(module, class_name)
    return provider_class
