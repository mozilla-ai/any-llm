"""Import shim: llama.cpp migrated to the config registry.

The provider is now a row in ``any_llm.providers.registry``; this module keeps
the historical deep-import path working.
"""

from any_llm.providers.registry import get_registry_provider_class

LlamacppProvider = get_registry_provider_class("llamacpp")

__all__ = ["LlamacppProvider"]
