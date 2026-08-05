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


class ProviderTier(StrEnum):
    """How far our support promise for a provider goes.

    A support promise, not a statement about code shape: a config-only registry
    row we hold a key for is still verified.
    """

    VERIFIED = "verified"
    """We hold an API key, integration tests run in CI, and we fix breakage."""

    COMMUNITY = "community"
    """Verified live by the contributor at PR time, then community-maintained.
    No CI key, so its integration tests skip in CI."""


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
    GMI = "gmi"
    GROQ = "groq"
    HUGGINGFACE = "huggingface"
    INCEPTION = "inception"
    KENARI = "kenari"
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
    EDENAI = "edenai"
    ZAI = "zai"
    TELNYX = "telnyx"

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


# The single source of truth for the verified tier: the providers CI actually
# holds keys for. `tests/unit/test_provider_tiers.py` asserts this matches the
# EXPECTED_PROVIDERS lists in .github/workflows/tests-integration.yaml, so the
# advertised tier cannot drift from which keys CI really has.
#
# Spelled as LLMProvider members rather than strings: promotion to verified
# requires an enum member anyway (the enum drives the integration test matrix),
# so a typo here fails at import instead of only in the parity test.
VERIFIED_LOCAL_PROVIDERS: frozenset[LLMProvider] = frozenset(
    {
        LLMProvider.OLLAMA,
        LLMProvider.LLAMACPP,
        LLMProvider.LLAMAFILE,
        LLMProvider.LMSTUDIO,
    }
)
"""Local servers CI stands up rather than authenticates against."""

VERIFIED_PROVIDERS: frozenset[LLMProvider] = VERIFIED_LOCAL_PROVIDERS | frozenset(
    {
        LLMProvider.ANTHROPIC,
        LLMProvider.AZUREOPENAI,
        LLMProvider.BEDROCK,
        LLMProvider.CEREBRAS,
        LLMProvider.COHERE,
        LLMProvider.DEEPSEEK,
        LLMProvider.FIREWORKS,
        LLMProvider.GEMINI,
        LLMProvider.GROQ,
        LLMProvider.INCEPTION,
        LLMProvider.MISTRAL,
        LLMProvider.MOONSHOT,
        LLMProvider.NEBIUS,
        LLMProvider.OPENAI,
        LLMProvider.OPENROUTER,
        LLMProvider.OTARI,
        LLMProvider.PORTKEY,
        LLMProvider.TOGETHER,
        LLMProvider.SAMBANOVA,
        LLMProvider.VOYAGE,
        LLMProvider.XAI,
        LLMProvider.ZAI,
        LLMProvider.MINIMAX,
    }
)


def get_provider_tier(provider_name: str) -> ProviderTier:
    """Return the support tier for a provider name."""
    if provider_name.strip().lower() in VERIFIED_PROVIDERS:
        return ProviderTier.VERIFIED
    return ProviderTier.COMMUNITY
