"""Import shim: Cascadia migrated to the config registry.

The provider is now a row in ``any_llm.providers.registry``; this module keeps
the historical deep-import path working.
"""

from any_llm.providers.registry import get_registry_provider_class

CascadiaProvider = get_registry_provider_class("cascadia")

__all__ = ["CascadiaProvider"]
