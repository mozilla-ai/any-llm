import logging
from collections.abc import Sequence
from typing import Any
from urllib.parse import urljoin

import httpx
from typing_extensions import override

from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.completion import CompletionParams
from any_llm.types.model import Model

logger = logging.getLogger(__name__)

CATALOG_PATH = "/catalog/models"
DEFAULT_TIMEOUT = 60.0


class GithubProvider(BaseOpenAIProvider):
    """Provider for the GitHub Models inference API.

    GitHub Models exposes OpenAI-compatible endpoints for chat completions
    and embeddings, backed by models from OpenAI, Meta, DeepSeek, and others.
    Auth uses a GitHub personal access token with ``models:read`` scope.
    """

    API_BASE = "https://models.github.ai/inference"
    ENV_API_KEY_NAME = "GITHUB_TOKEN"
    ENV_API_BASE_NAME = "GITHUB_MODELS_API_BASE"
    PROVIDER_NAME = "github"
    PROVIDER_DOCUMENTATION_URL = "https://docs.github.com/en/github-models"

    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_RESPONSES = False
    SUPPORTS_MODERATION = False
    SUPPORTS_BATCH = False

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        """GitHub Models only accepts ``max_tokens``, not ``max_completion_tokens``."""
        converted_params = BaseOpenAIProvider._convert_completion_params(params, **kwargs)
        if "max_completion_tokens" in converted_params:
            converted_params["max_tokens"] = converted_params.pop("max_completion_tokens")
        return converted_params

    def _get_catalog_url(self) -> str:
        """Derive the catalog URL from the configured base URL.

        The catalog endpoint lives at ``/catalog/models`` on the same origin
        as the inference base URL, so we strip the path from ``base_url``
        and append the catalog path.
        """
        base = str(self.client.base_url).rstrip("/")
        origin = base.split("/inference")[0] if "/inference" in base else base
        return urljoin(origin + "/", CATALOG_PATH.lstrip("/"))

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        """Fetch available models from the GitHub Models catalog.

        The catalog lives at a separate path (``/catalog/models``) outside
        the inference base URL, so we call it directly with ``httpx``
        instead of going through the OpenAI client.
        """
        catalog_url = self._get_catalog_url()
        headers = {
            "Authorization": f"Bearer {self.client.api_key}",
            "Accept": "application/vnd.github+json",
        }

        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(catalog_url, headers=headers, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            data = response.json()

        return self._convert_list_models_response(data)

    @staticmethod
    @override
    def _convert_list_models_response(response: Any) -> Sequence[Model]:
        """Map GitHub catalog entries to the canonical ``Model`` type.

        The catalog returns objects with ``id``, ``name``, ``publisher``, etc.
        We map ``id`` to ``Model.id`` and ``publisher`` to ``Model.owned_by``.
        Malformed entries (non-dict or missing ``id``) are skipped with a warning.
        """
        models: list[Model] = []
        items = response if isinstance(response, list) else []
        for item in items:
            if not isinstance(item, dict):
                logger.warning("Skipping non-dict catalog entry: %s", type(item).__name__)
                continue
            model_id = item.get("id")
            if not model_id:
                logger.warning("Skipping catalog entry with missing 'id': %s", item)
                continue
            models.append(
                Model(
                    id=model_id,
                    created=0,
                    object="model",
                    owned_by=item.get("publisher", "unknown"),
                )
            )
        return models
