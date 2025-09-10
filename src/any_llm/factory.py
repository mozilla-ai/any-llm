import importlib
import warnings
from pathlib import Path

from any_llm.config import ClientConfig
from any_llm.constants import ProviderName
from any_llm.exceptions import UnsupportedProviderError
from any_llm.provider import Provider
from any_llm.types.provider import ProviderMetadata


class ProviderFactory:
    """Factory to dynamically load provider instances based on the naming conventions."""

    PROVIDERS_DIR = Path(__file__).parent / "providers"

    @classmethod
    def create_provider(cls, provider_key: str | ProviderName, config: ClientConfig) -> Provider:
        """Dynamically load and create an instance of a provider based on the naming convention."""
        provider_key = ProviderName.from_string(provider_key).value

        provider_class_name = f"{provider_key.capitalize()}Provider"
        provider_module_name = f"{provider_key}"

        module_path = f"any_llm.providers.{provider_module_name}"

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            msg = f"Could not import module {module_path}: {e!s}. Please ensure the provider is supported by doing ProviderFactory.get_supported_providers()"
            raise ImportError(msg) from e

        provider_class: type[Provider] = getattr(module, provider_class_name)
        return provider_class(config=config)

    @classmethod
    def get_provider_class(cls, provider_key: str | ProviderName) -> type[Provider]:
        """Get the provider class without instantiating it.

        Args:
            provider_key: The provider key (e.g., 'anthropic', 'openai')

        Returns:
            The provider class

        """
        provider_key = ProviderName.from_string(provider_key).value

        provider_class_name = f"{provider_key.capitalize()}Provider"
        provider_module_name = f"{provider_key}"

        module_path = f"any_llm.providers.{provider_module_name}"

        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            msg = f"Could not import module {module_path}: {e!s}. Please ensure the provider is supported by doing ProviderFactory.get_supported_providers()"
            raise ImportError(msg) from e

        provider_class: type[Provider] = getattr(module, provider_class_name)
        return provider_class

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get a list of supported provider keys."""
        return [provider.value for provider in ProviderName]

    @classmethod
    def get_all_provider_metadata(cls) -> list[ProviderMetadata]:
        """Get metadata for all supported providers.

        Returns:
            List of dictionaries containing provider metadata

        """
        providers: list[ProviderMetadata] = []
        for provider_key in cls.get_supported_providers():
            provider_class = cls.get_provider_class(provider_key)
            metadata = provider_class.get_provider_metadata()
            providers.append(metadata)

        # Sort providers by name
        providers.sort(key=lambda x: x.name)
        return providers

    @classmethod
    def get_provider_enum(cls, provider_key: str) -> ProviderName:
        """Convert a string provider key to a ProviderName enum."""
        try:
            return ProviderName(provider_key)
        except ValueError as e:
            supported = [provider.value for provider in ProviderName]
            raise UnsupportedProviderError(provider_key, supported) from e

    @classmethod
    def split_model_provider(cls, model: str) -> tuple[ProviderName, str]:
        """Extract the provider key from the model identifier.

        Supports both new format 'provider:model' (e.g., 'mistral:mistral-small')
        and legacy format 'provider/model' (e.g., 'mistral/mistral-small').

        The legacy format will be deprecated in version 1.0.
        """
        colon_index = model.find(":")
        slash_index = model.find("/")

        # Determine which delimiter comes first
        if colon_index != -1 and (slash_index == -1 or colon_index < slash_index):
            # The colon came first, so it's using the new syntax.
            provider, model_name = model.split(":", 1)
        elif slash_index != -1:
            # Slash comes first, so it's the legacy syntax
            warnings.warn(
                f"Model format 'provider/model' is deprecated and will be removed in version 1.0. "
                f"Please use 'provider:model' format instead. Got: '{model}'",
                DeprecationWarning,
                stacklevel=3,
            )
            provider, model_name = model.split("/", 1)
        else:
            msg = f"Invalid model format. Expected 'provider:model' or 'provider/model', got '{model}'"
            raise ValueError(msg)

        if not provider or not model_name:
            msg = f"Invalid model format. Expected 'provider:model' or 'provider/model', got '{model}'"
            raise ValueError(msg)
        return cls.get_provider_enum(provider), model_name
