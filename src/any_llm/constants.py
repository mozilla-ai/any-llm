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
    AZUREANTHROPIC = "azureanthropic"
    AZUREOPENAI = "azureopenai"
    ATLASCLOUD = "atlascloud"
    CASCADIA = "cascadia"
    CEREBRAS = "cerebras"
    COHERE = "cohere"
    DATABRICKS = "databricks"
    DEEPSEEK = "deepseek"
    FIREWORKS = "fireworks"
    GEMINI = "gemini"
    GITHUB = "github"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    INCEPTION = "inception"
    LLAMA = "llama"
    LMSTUDIO = "lmstudio"
    LLAMAFILE = "llamafile"
    LLAMACPP = "llamacpp"
    MISTRAL = "mistral"
    MOONSHOT = "moonshot"
    MZAI = "mzai"
    NEOSANTARA = "neosantara"
    NEBIUS = "nebius"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OTARI = "otari"
    OPENROUTER = "openrouter"
    PORTKEY = "portkey"
    QINIU = "qiniu"
    REQUESTY = "requesty"
    SAMBANOVA = "sambanova"
    SAGEMAKER = "sagemaker"
    TOGETHER = "together"
    VERTEXAI = "vertexai"
    VERTEXAIANTHROPIC = "vertexaianthropic"
    VLLM = "vllm"
    VOYAGE = "voyage"
    WATSONX = "watsonx"
    XAI = "xai"
    PERPLEXITY = "perplexity"
    MINIMAX = "minimax"
    DASHSCOPE = "dashscope"
    DEEPINFRA = "deepinfra"
    ZAI = "zai"

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
