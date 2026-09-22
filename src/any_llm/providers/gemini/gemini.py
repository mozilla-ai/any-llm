import os
from collections.abc import AsyncIterator
from typing import Any, ClassVar

from google import genai
from google.genai import types
from typing_extensions import override

from any_llm.exceptions import MissingApiKeyError, UnsupportedParameterError
from any_llm.types.files import FileDeleted, FileInput, FileMetadata, FileOperation, FilePage
from any_llm.types.responses import Response, ResponsesParams, ResponseStreamEvent
from any_llm.utils.aio import aclose_quietly

from .base import GoogleProvider
from .files import delete_file, list_files, retrieve_file, upload_file


class GeminiProvider(GoogleProvider):
    """Gemini Provider using the Google GenAI Developer API."""

    PROVIDER_NAME = "gemini"
    PROVIDER_DOCUMENTATION_URL = "https://ai.google.dev/gemini-api/docs"
    ENV_API_KEY_NAME = "GEMINI_API_KEY/GOOGLE_API_KEY"
    ENV_API_BASE_NAME = "GOOGLE_GEMINI_BASE_URL"
    SUPPORTS_RESPONSES = True
    # Downloads stay out: files.download buffers a whole file in memory and reports neither
    # the response status nor its headers. Honoring the streamed-download contract would mean
    # bypassing the SDK and requesting files/<id>:download?alt=media over raw HTTP, which is a
    # larger commitment than this change makes.
    SUPPORTED_FILE_OPERATIONS: ClassVar[frozenset[FileOperation]] = frozenset({"upload", "list", "retrieve", "delete"})

    _interactions_api_version: str | None

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
        # Follow the client's configured version, leaving None to the SDK default of
        # v1beta. Interactions is served on both v1 and v1beta, but v1 does not carry
        # preview models, so pinning it here would reject models completion() accepts.
        # https://ai.google.dev/gemini-api/docs/api-versions
        self._interactions_api_version = configured_api_version

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
        timeout = kwargs.pop("timeout", None)
        create_kwargs = {name: value for name, value in kwargs.items() if value is not None}
        unsupported_parameters = create_kwargs.keys() - {"extra_headers", "extra_query"}
        if unsupported_parameters:
            parameter_name = min(unsupported_parameters)
            raise UnsupportedParameterError(parameter_name, self.PROVIDER_NAME)

        # Vertex shares this package without requiring the Interactions SDK.
        from .interactions import convert_interaction_to_response, convert_responses_params

        create_kwargs = (
            convert_responses_params(
                params,
                self.PROVIDER_NAME,
                api_version=self._interactions_api_version,
            )
            | create_kwargs
        )
        if timeout is not None:
            create_kwargs["timeout"] = timeout
        if params.stream:
            return self._create_interaction_stream(create_kwargs, model=params.model)
        interaction = await self.client.aio.interactions.create(**create_kwargs)
        return convert_interaction_to_response(interaction, fallback_model=params.model)

    async def _create_interaction_stream(
        self, create_kwargs: dict[str, Any], *, model: str
    ) -> AsyncIterator[ResponseStreamEvent]:
        from .interactions_stream import convert_interaction_stream

        stream = await self.client.aio.interactions.create(**create_kwargs)
        converted_stream = convert_interaction_stream(stream, model=model)
        try:
            async for event in converted_stream:
                yield event
        finally:
            await aclose_quietly(converted_stream)

    @override
    async def _aupload_file(
        self,
        file: FileInput,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
        purpose: str | None = None,
        expires_in: int | None = None,
        **kwargs: Any,
    ) -> FileMetadata:
        return await upload_file(self.client, file, filename, mime_type, purpose, expires_in, kwargs)

    @override
    async def _alist_files(
        self, *, limit: int | None = None, cursor: str | None = None, purpose: str | None = None, **kwargs: Any
    ) -> FilePage:
        return await list_files(self.client, limit, cursor, purpose, kwargs)

    @override
    async def _aretrieve_file(self, file_id: str, **kwargs: Any) -> FileMetadata:
        return await retrieve_file(self.client, file_id, kwargs, self._unified_exceptions)

    @override
    async def _adelete_file(self, file_id: str, **kwargs: Any) -> FileDeleted:
        return await delete_file(self.client, file_id, kwargs, self._unified_exceptions)
