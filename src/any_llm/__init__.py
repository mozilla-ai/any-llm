from importlib.metadata import PackageNotFoundError, version

from any_llm.any_llm import AnyLLM
from any_llm.api import acompletion, aembedding, alist_models, aresponses, completion, embedding, list_models, responses
from any_llm.constants import LLMProvider

try:
    __version__ = version("any-llm-sdk")
except PackageNotFoundError:
    # In the case of local development
    # i.e., running directly from the source directory without package being installed
    __version__ = "0.0.0-dev"


__all__ = [
    "AnyLLM",
    "LLMProvider",
    "acompletion",
    "aembedding",
    "alist_models",
    "aresponses",
    "completion",
    "embedding",
    "list_models",
    "responses",
]
