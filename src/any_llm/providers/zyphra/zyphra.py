from collections.abc import Sequence
from datetime import datetime
from typing import Any

import httpx
from typing_extensions import override

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import CompletionParams
from any_llm.types.model import Model


class ZyphraProvider(BaseOpenAIProvider):
    API_BASE = "https://api.zyphracloud.com/api/v1"
    ENV_API_KEY_NAME = "ZYPHRA_API_KEY"
    ENV_API_BASE_NAME = "ZYPHRA_API_BASE"
    PROVIDER_NAME = "zyphra"
    PROVIDER_DOCUMENTATION_URL = "https://zyphra.com"

    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_LIST_MODELS = True

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        converted = BaseOpenAIProvider._convert_completion_params(params, **kwargs)
        # Zyphra's backend does not accept reasoning_effort="none"; drop it so the
        # request omits the field entirely. "auto" is already normalized upstream.
        if converted.get("reasoning_effort") == "none":
            converted.pop("reasoning_effort", None)
        return converted

    @staticmethod
    def _zyphra_item_to_model(item: dict[str, Any]) -> Model:
        # Zyphra's /models response is not OpenAI-shaped: it's a bare JSON array
        # whose items use camelCase keys (modelId, organization, releaseDate) and
        # omit OpenAI's required object/created/owned_by fields. Map them here.
        release_date = item.get("releaseDate")
        created = 0
        if isinstance(release_date, str):
            try:
                created = int(datetime.fromisoformat(release_date).timestamp())
            except ValueError:
                created = 0
        return Model(
            id=item["modelId"],
            object="model",
            created=created,
            owned_by=item.get("organization", "zyphra"),
        )

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        base_url = str(self.client.base_url).rstrip("/")
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            resp = await http_client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {self.client.api_key}"},
            )
            resp.raise_for_status()
            payload = resp.json()
        return [self._zyphra_item_to_model(item) for item in payload]
