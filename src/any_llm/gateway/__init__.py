import warnings

warnings.warn(
    "any_llm.gateway is deprecated and will be removed on May 18, 2026. "
    "Migrate to the standalone package at https://github.com/mozilla-ai/gateway",
    DeprecationWarning,
    stacklevel=2,
)

from any_llm import __version__

__all__ = ["__version__"]
