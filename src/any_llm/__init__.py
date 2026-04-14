from importlib.metadata import PackageNotFoundError, version

from any_llm.any_llm import AnyLLM
from any_llm.api import (
    acancel_batch,
    acompletion,
    acreate_batch,
    aembedding,
    alist_batches,
    alist_models,
    amessages,
    aresponses,
    aretrieve_batch,
    cancel_batch,
    completion,
    create_batch,
    embedding,
    list_batches,
    list_models,
    messages,
    responses,
    retrieve_batch,
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
    "acancel_batch",
    "acompletion",
    "acreate_batch",
    "aembedding",
    "alist_batches",
    "alist_models",
    "amessages",
    "aresponses",
    "aretrieve_batch",
    "cancel_batch",
    "completion",
    "create_batch",
    "embedding",
    "list_batches",
    "list_models",
    "messages",
    "responses",
    "retrieve_batch",
]
