from __future__ import annotations

from typing import TYPE_CHECKING, Any

from typing_extensions import override

from .base import BaseAnthropicProvider

MISSING_PACKAGES_ERROR = None
try:
    from anthropic import AsyncAnthropic
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import Sequence

    from any_llm.types.model import Model


class AnthropicProvider(BaseAnthropicProvider):
    """
    Anthropic Provider using enhanced Provider framework.

    Handles conversion between OpenAI format and Anthropic's native format.
    """

    PROVIDER_NAME = "anthropic"
    ENV_API_KEY_NAME = "ANTHROPIC_API_KEY"
    PROVIDER_DOCUMENTATION_URL = "https://docs.anthropic.com/en/home"

    SUPPORTS_LIST_MODELS = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncAnthropic

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=api_base,
            **kwargs,
        )

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        models_list = await self.client.models.list(**kwargs)
        return self._convert_list_models_response(models_list.data)
