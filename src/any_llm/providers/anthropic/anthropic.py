from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar

from typing_extensions import override

from any_llm.types.files import AsyncFileDownload, FileDeleted, FileInput, FileMetadata, FileOperation, FilePage

from .base import BaseAnthropicProvider
from .files import convert_metadata, list_files, reject_unsupported, request_options, upload_file, validate_file_id

MISSING_PACKAGES_ERROR = None
try:
    from anthropic import AsyncAnthropic
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from any_llm.types.model import Model


class AnthropicProvider(BaseAnthropicProvider):
    """
    Anthropic Provider using enhanced Provider framework.

    Handles conversion between OpenAI format and Anthropic's native format.
    """

    PROVIDER_NAME = "anthropic"
    ENV_API_KEY_NAME = "ANTHROPIC_API_KEY"
    ENV_API_BASE_NAME = "ANTHROPIC_BASE_URL"
    PROVIDER_DOCUMENTATION_URL = "https://docs.anthropic.com/en/home"

    SUPPORTS_LIST_MODELS = True

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    SUPPORTED_FILE_OPERATIONS: ClassVar[frozenset[FileOperation]] = frozenset(
        {"upload", "list", "retrieve", "download", "delete"}
    )

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
        client, options = request_options(self.client, kwargs)
        reject_unsupported(kwargs)
        validate_file_id(file_id)
        result = await client.files.retrieve_metadata(file_id, **options)
        return convert_metadata(result)

    @override
    async def _adelete_file(self, file_id: str, **kwargs: Any) -> FileDeleted:
        client, options = request_options(self.client, kwargs)
        reject_unsupported(kwargs)
        validate_file_id(file_id)
        result = await client.files.delete(file_id, **options)
        return FileDeleted.model_validate(result.model_dump(exclude_unset=True))

    @override
    @asynccontextmanager
    async def _adownload_file(
        self, file_id: str, *, chunk_size: int, **kwargs: Any
    ) -> AsyncIterator[AsyncFileDownload]:
        client, options = request_options(self.client, kwargs)
        reject_unsupported(kwargs)
        validate_file_id(file_id)
        async with client.files.with_streaming_response.download(file_id, **options) as response:
            yield AsyncFileDownload(
                status_code=response.status_code,
                headers=response.headers.copy(),
                chunks=response.iter_bytes(chunk_size),
            )
