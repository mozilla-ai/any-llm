from __future__ import annotations

from contextlib import ExitStack, asynccontextmanager
from os import PathLike
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

from typing_extensions import override

from any_llm._files import FilesMixin
from any_llm.exceptions import InvalidRequestError, ProviderError, UnsupportedParameterError
from any_llm.types.files import AsyncFileDownload, FileDeleted, FileInput, FileMetadata, FilePage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from openai import AsyncOpenAI
    from openai.types import FileObject, FilePurpose


class OpenAIFileMethods(FilesMixin):
    """Opt-in Files implementation using the OpenAI SDK.

    Capabilities belong to the concrete providers, not the shared OpenAI base.
    """

    client: AsyncOpenAI

    def _file_request_options(
        self, kwargs: dict[str, Any], *, upload: bool = False
    ) -> tuple[AsyncOpenAI, dict[str, Any]]:
        client = self.client
        options: dict[str, Any] = {key: kwargs.pop(key) for key in ("timeout", "extra_headers") if key in kwargs}
        retries = kwargs.pop("max_retries", 0 if upload else None)
        if retries is not None:
            client = client.with_options(max_retries=retries)
        if kwargs:
            raise UnsupportedParameterError(", ".join(sorted(kwargs)), self.PROVIDER_NAME)
        return client, options

    def _validate_file_id(self, file_id: str) -> None:
        if (
            not file_id
            or file_id in {".", ".."}
            or any(character in file_id for character in "/\\?#%")
            or any(character.isspace() for character in file_id)
        ):
            message = "A nonempty provider file ID without path separators, URL delimiters, or whitespace is required"
            raise InvalidRequestError(message, provider_name=self.PROVIDER_NAME)

    @staticmethod
    def _convert_file_metadata(result: FileObject) -> FileMetadata:
        data = result.model_dump(exclude_unset=True)
        data["size_bytes"] = data.pop("bytes", None)
        return FileMetadata.model_validate(data)

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
        if not isinstance(purpose, str) or not purpose.strip():
            message = "purpose is required for file uploads"
            raise InvalidRequestError(message, provider_name=self.PROVIDER_NAME)
        client, options = self._file_request_options(kwargs, upload=True)
        if expires_in is not None:
            if isinstance(expires_in, bool) or not isinstance(expires_in, int) or expires_in <= 0:
                message = "expires_in must be a positive integer number of seconds"
                raise InvalidRequestError(message, provider_name=self.PROVIDER_NAME)
            options["expires_after"] = {"anchor": "created_at", "seconds": expires_in}
        content: bytes | IO[bytes]
        with ExitStack() as stack:
            if isinstance(file, (str, PathLike)):
                path = Path(file)
                try:
                    content = stack.enter_context(path.open("rb"))  # noqa: ASYNC230 (SDK multipart encoding uses synchronous handles)
                except OSError as exc:
                    message = f"Cannot open upload path {str(path)!r}: {exc.strerror or exc}"
                    raise InvalidRequestError(
                        message, original_exception=exc, provider_name=self.PROVIDER_NAME
                    ) from exc
                filename = filename or path.name
            else:
                content = file
            result = await client.files.create(
                file=(filename or "upload", content, mime_type or "application/octet-stream"),
                purpose=cast("FilePurpose", purpose),
                **options,
            )
        return self._convert_file_metadata(result)

    @override
    async def _alist_files(
        self, *, limit: int | None = None, cursor: str | None = None, purpose: str | None = None, **kwargs: Any
    ) -> FilePage:
        order = kwargs.pop("order", None)
        if order is not None and order not in {"asc", "desc"}:
            message = "order must be 'asc' or 'desc'"
            raise InvalidRequestError(message, provider_name=self.PROVIDER_NAME)
        client, options = self._file_request_options(kwargs)
        if order is not None:
            options["order"] = order
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                message = "limit must be a positive integer"
                raise InvalidRequestError(message, provider_name=self.PROVIDER_NAME)
            options["limit"] = limit
        if cursor is not None:
            self._validate_file_id(cursor)
            options["after"] = cursor
        if purpose is not None:
            if not isinstance(purpose, str):
                message = "purpose must be a string"
                raise InvalidRequestError(message, provider_name=self.PROVIDER_NAME)
            options["purpose"] = purpose
        page = await client.files.list(**options)
        if page.has_more is None or (page.has_more and not page.data):
            message = "Files response has missing or inconsistent pagination information"
            raise ProviderError(message, provider_name=self.PROVIDER_NAME)
        next_cursor = page.data[-1].id if page.has_more else None
        extras = {
            key: value
            for key, value in page.model_dump(exclude_unset=True).items()
            if key not in {"data", "next_cursor"}
        }
        return FilePage(
            data=[self._convert_file_metadata(item) for item in page.data], next_cursor=next_cursor, **extras
        )

    @override
    async def _aretrieve_file(self, file_id: str, **kwargs: Any) -> FileMetadata:
        self._validate_file_id(file_id)
        client, options = self._file_request_options(kwargs)
        return self._convert_file_metadata(await client.files.retrieve(file_id, **options))

    @override
    async def _adelete_file(self, file_id: str, **kwargs: Any) -> FileDeleted:
        self._validate_file_id(file_id)
        client, options = self._file_request_options(kwargs)
        result = await client.files.delete(file_id, **options)
        return FileDeleted.model_validate(result.model_dump(exclude_unset=True))

    @override
    @asynccontextmanager
    async def _adownload_file(
        self, file_id: str, *, chunk_size: int, **kwargs: Any
    ) -> AsyncIterator[AsyncFileDownload]:
        self._validate_file_id(file_id)
        client, options = self._file_request_options(kwargs)
        async with client.files.with_streaming_response.content(file_id, **options) as response:
            yield AsyncFileDownload(
                status_code=response.status_code,
                headers=response.headers.copy(),
                chunks=response.iter_bytes(chunk_size),
            )
