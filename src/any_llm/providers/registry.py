"""Config registry for OpenAI-compatible gateway providers.

Config-only providers (no method overrides, only class attributes on top of
``BaseOpenAIProvider``) are described here as data rows instead of code folders.
The loader in ``AnyLLM`` checks this registry before falling through to the
folder-per-provider import convention, so a registry row resolves through
``AnyLLM.create(...)``, ``AnyLLM.get_provider_class(...)`` and the metadata
APIs like any folder-based provider.

Adding a community gateway means adding one row (plus live verification
evidence in the PR). Removing a dead gateway means deleting its row. See the
provider policy in https://github.com/mozilla-ai/any-llm/issues/1197.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from any_llm.providers.openai.base import BaseOpenAIProvider


@dataclass(frozen=True)
class OpenAICompatibleProviderConfig:
    """Everything a config-only OpenAI-compatible provider needs, as data.

    Capability flags default to the conservative gateway baseline (completion,
    streaming and model listing only); rows opt in to anything beyond that.
    A gateway that needs behavior (custom auth, request/response translation,
    param remaps) does not belong in the registry and keeps a code folder.
    """

    name: str
    api_base: str
    env_api_key_name: str
    provider_documentation_url: str
    env_api_base_name: str | None = None
    supports_completion: bool = True
    supports_completion_streaming: bool = True
    supports_completion_reasoning: bool = False
    supports_completion_image: bool = False
    supports_completion_pdf: bool = False
    supports_embedding: bool = False
    supports_moderation: bool = False
    supports_list_models: bool = True
    supports_responses: bool = False
    supports_batch: bool = False
    supports_image_generation: bool = False
    supports_rerank: bool = False


PROVIDER_REGISTRY: dict[str, OpenAICompatibleProviderConfig] = {
    "atlascloud": OpenAICompatibleProviderConfig(
        name="atlascloud",
        api_base="https://api.atlascloud.ai/v1",
        env_api_key_name="ATLASCLOUD_API_KEY",
        env_api_base_name="ATLASCLOUD_API_BASE",
        provider_documentation_url="https://www.atlascloud.ai/docs",
        supports_completion_reasoning=True,
    ),
}

_class_cache: dict[str, type[BaseOpenAIProvider]] = {}


def get_registry_config(name: str) -> OpenAICompatibleProviderConfig | None:
    """Return the registry row for ``name``, or None if it is not registered."""
    return PROVIDER_REGISTRY.get(name.strip().lower())


def get_registry_provider_class(name: str) -> type[BaseOpenAIProvider]:
    """Return the provider class for a registry row, minting it on first use.

    Classes are cached per row so repeated lookups return the same class object
    (identity checks like ``get_provider_class(name) is AtlascloudProvider``
    keep working). Raises KeyError for names that are not in the registry.
    """
    key = name.strip().lower()
    config = PROVIDER_REGISTRY[key]
    if key not in _class_cache:
        _class_cache[key] = _build_provider_class(config)
    return _class_cache[key]


def _build_provider_class(config: OpenAICompatibleProviderConfig) -> type[BaseOpenAIProvider]:
    attrs: dict[str, Any] = {
        "PROVIDER_NAME": config.name,
        "PROVIDER_DOCUMENTATION_URL": config.provider_documentation_url,
        "ENV_API_KEY_NAME": config.env_api_key_name,
        "ENV_API_BASE_NAME": config.env_api_base_name,
        "API_BASE": config.api_base,
        "SUPPORTS_COMPLETION": config.supports_completion,
        "SUPPORTS_COMPLETION_STREAMING": config.supports_completion_streaming,
        "SUPPORTS_COMPLETION_REASONING": config.supports_completion_reasoning,
        "SUPPORTS_COMPLETION_IMAGE": config.supports_completion_image,
        "SUPPORTS_COMPLETION_PDF": config.supports_completion_pdf,
        "SUPPORTS_EMBEDDING": config.supports_embedding,
        "SUPPORTS_MODERATION": config.supports_moderation,
        "SUPPORTS_LIST_MODELS": config.supports_list_models,
        "SUPPORTS_RESPONSES": config.supports_responses,
        "SUPPORTS_BATCH": config.supports_batch,
        "SUPPORTS_IMAGE_GENERATION": config.supports_image_generation,
        "SUPPORTS_RERANK": config.supports_rerank,
    }
    # The class name follows the same convention as folder-based providers so
    # metadata.class_name is stable across a migration.
    class_name = f"{config.name.capitalize()}Provider"
    return cast("type[BaseOpenAIProvider]", type(class_name, (BaseOpenAIProvider,), attrs))
