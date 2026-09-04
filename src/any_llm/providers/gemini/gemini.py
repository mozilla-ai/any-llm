import os
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types
from typing_extensions import override

from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.types.responses import Response, ResponsesParams, ResponseStreamEvent

from .base import GoogleProvider
from .interactions import convert_interaction_to_response, convert_responses_params
from .interactions_stream import convert_interaction_stream


class GeminiProvider(GoogleProvider):
    """Gemini Provider using the Google GenAI Developer API."""

    PROVIDER_NAME = "gemini"
    PROVIDER_DOCUMENTATION_URL = "https://ai.google.dev/gemini-api/docs"
    ENV_API_KEY_NAME = "GEMINI_API_KEY/GOOGLE_API_KEY"
    ENV_API_BASE_NAME = "GOOGLE_GEMINI_BASE_URL"
    SUPPORTS_RESPONSES = True

    _interactions_api_version: str

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise MissingApiKeyError(self.PROVIDER_NAME, self.ENV_API_KEY_NAME)
        return api_key

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        http_options = kwargs.get("http_options")
        if isinstance(http_options, dict):
            configured_api_version = http_options.get("api_version")
        elif isinstance(http_options, types.HttpOptions):
            configured_api_version = http_options.api_version
        else:
            configured_api_version = None
        # Interactions is GA in v1, while the SDK defaults the shared Gemini
        # Developer API client to v1beta for generateContent preview features.
        # https://ai.google.dev/gemini-api/docs/api-versions
        self._interactions_api_version = configured_api_version or "v1"

        if api_base:
            http_options = kwargs.pop("http_options", None)
            if http_options is None:
                http_options = types.HttpOptions(base_url=api_base)
            elif isinstance(http_options, dict):
                http_options.setdefault("base_url", api_base)
            elif isinstance(http_options, types.HttpOptions) and http_options.base_url is None:
                http_options.base_url = api_base
            kwargs["http_options"] = http_options

        # Ensure timeout is correctly configured if present.
        if (timeout := kwargs.pop("timeout", None)) is not None:
            GoogleProvider._merge_timeout_into_http_options(timeout, kwargs)

        self.client = genai.Client(api_key=api_key, **kwargs)

    @override
    async def _aresponses(
        self, params: ResponsesParams, **kwargs: Any
    ) -> Response | AsyncIterator[ResponseStreamEvent]:
        if kwargs.pop("extra_body", None) is not None:
            parameter_name = "extra_body"
            raise UnsupportedParameterError(parameter_name, self.PROVIDER_NAME)
        create_kwargs = convert_responses_params(
            params,
            self.PROVIDER_NAME,
            api_version=self._interactions_api_version,
        )
        if (timeout := kwargs.get("timeout")) is not None:
            create_kwargs["timeout"] = timeout
        if params.stream:
            stream = await self.client.aio.interactions.create(**create_kwargs)
            return convert_interaction_stream(stream, model=params.model)
        interaction = await self.client.aio.interactions.create(**create_kwargs)
        return convert_interaction_to_response(interaction)
