import builtins
from enum import StrEnum

from any_llm.exceptions import UnsupportedProviderError

INSIDE_NOTEBOOK = hasattr(builtins, "__IPYTHON__")

REASONING_FIELD_NAMES = [
    "reasoning_content",
    "thinking",
    "think",
    "chain_of_thought",
]


class LLMProvider(StrEnum):
    """String enum for supported providers."""

    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    AZURE = "azure"
    AZUREOPENAI = "azureopenai"
    CEREBRAS = "cerebras"
    COHERE = "cohere"
    DATABRICKS = "databricks"
    DEEPSEEK = "deepseek"
    FIREWORKS = "fireworks"
    GEMINI = "gemini"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    INCEPTION = "inception"
    LLAMA = "llama"
    LMSTUDIO = "lmstudio"
    LLAMAFILE = "llamafile"
    LLAMACPP = "llamacpp"
    MISTRAL = "mistral"
    MOONSHOT = "moonshot"
    NEBIUS = "nebius"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    PLATFORM = "platform"
    PORTKEY = "portkey"
    SAMBANOVA = "sambanova"
    SAGEMAKER = "sagemaker"
    TOGETHER = "together"
    VERTEXAI = "vertexai"
    VOYAGE = "voyage"
    WATSONX = "watsonx"
    XAI = "xai"
    PERPLEXITY = "perplexity"
    MINIMAX = "minimax"
    ZAI = "zai"
    GATEWAY = "gateway"

    @classmethod
    def from_string(cls, value: "str | LLMProvider") -> "LLMProvider":
        """Convert a string to a ProviderName enum."""
        if isinstance(value, cls):
            return value

        formatted_value = value.strip().lower()
        try:
            return cls(formatted_value)
        except ValueError as exc:
            supported = [provider.value for provider in cls]
            raise UnsupportedProviderError(value, supported) from exc
