from __future__ import annotations

from typing import TYPE_CHECKING

from .utils import get_provider_key

if TYPE_CHECKING:
    from any_llm.any_llm import AnyLLM

ANY_API_URL = "http://localhost:8000/api/v1"


class AnyAPIProvider:
    @staticmethod
    def prepare(
        provider_class: type[AnyLLM],
        any_api_key: str,
        project_id: str,
        api_base: str | None = None,
        **kwargs
    ) -> AnyLLM:
        provider_key = get_provider_key(
            any_api_key=any_api_key,
            project_id=project_id,
            provider=provider_class,
            any_api_url=ANY_API_URL
        )

        return provider_class(api_key=provider_key, api_base=api_base, **kwargs)
