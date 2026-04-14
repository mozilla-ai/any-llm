from importlib.metadata import PackageNotFoundError, version

from any_llm.any_llm import AnyLLM
from any_llm.api import (
    acompletion,
    aembedding,
    aimage_generation,
    alist_models,
    amessages,
    aresponses,
    completion,
    embedding,
    image_generation,
    list_models,
    messages,
    responses,
)
from any_llm.constants import LLMProvider
from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError,
    ContentFilterError,
    ContentFilterFinishReasonError,
    ContextLengthExceededError,
    InvalidRequestError,
    LengthFinishReasonError,
    MissingApiKeyError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    UnsupportedParameterError,
    UnsupportedProviderError,
)
from any_llm.types.completion import ParsedChatCompletion, ParsedChatCompletionMessage, ParsedChoice

try:
    __version__ = version("any-llm-sdk")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"


__all__ = [
    "AnyLLM",
    "AnyLLMError",
    "AuthenticationError",
    "ContentFilterError",
    "ContentFilterFinishReasonError",
    "ContextLengthExceededError",
    "InvalidRequestError",
    "LLMProvider",
    "LengthFinishReasonError",
    "MissingApiKeyError",
    "ModelNotFoundError",
    "ParsedChatCompletion",
    "ParsedChatCompletionMessage",
    "ParsedChoice",
    "ProviderError",
    "RateLimitError",
    "UnsupportedParameterError",
    "UnsupportedProviderError",
    "acompletion",
    "aembedding",
    "aimage_generation",
    "alist_models",
    "amessages",
    "aresponses",
    "completion",
    "embedding",
    "image_generation",
    "list_models",
    "messages",
    "responses",
]
