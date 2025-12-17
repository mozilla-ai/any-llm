"""Deprecated package redirect for any-llm-sdk.

DEPRECATION NOTICE
==================

The 'any-llm-sdk' package has been renamed to 'any-llm'.

Please update your dependencies:
- Old: pip install any-llm-sdk
- New: pip install any-llm

This package now simply re-exports everything from 'any-llm' for backwards compatibility.
This stub package will be maintained for a transition period, but all new development
happens in the 'any-llm' package.

Update your imports (though both will work):
- from any_llm import completion  # Works the same way
"""

import warnings

from any_llm import *  # noqa: F403
from any_llm import __version__  # type: ignore[attr-defined]

# Show deprecation warning when the package is imported
warnings.warn(
    "The 'any-llm-sdk' package has been renamed to 'any-llm'. "
    "Please update your dependencies to use 'pip install any-llm' instead. "
    "This stub package will be maintained for backwards compatibility but may be removed in the future.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AnyLLM",
    "LLMProvider",
    "__version__",
    "acompletion",
    "aembedding",
    "alist_models",
    "aresponses",
    "completion",
    "embedding",
    "list_models",
    "responses",
]
