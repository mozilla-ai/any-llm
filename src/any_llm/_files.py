from collections.abc import AsyncGenerator, AsyncIterator, Iterator
from contextlib import AsyncExitStack, aclosing, asynccontextmanager, contextmanager
from typing import Any, ClassVar, cast

from any_llm.constants import INSIDE_NOTEBOOK
from any_llm.exceptions import InvalidRequestError
from any_llm.types.files import (
    AsyncFileDownload,
    FileDeleted,
    FileDownload,
    FileInput,
    FileMetadata,
    FileOperation,
    FilePage,
)
from any_llm.utils.aio import _async_source_to_sync_iter, run_async_in_sync
from any_llm.utils.exception_handler import _handle_exception, handle_exceptions


class FilesMixin:
    """Public provider file operations and unsupported-provider defaults."""

    PROVIDER_NAME: str

    SUPPORTED_FILE_OPERATIONS: ClassVar[frozenset[FileOperation]] = frozenset()

    @handle_exceptions(file_operation=True)
    async def aupload_file(
        self, file: FileInput, *, filename: str | None = None, mime_type: str | None = None, **kwargs: Any
    ) -> FileMetadata:
        """Upload a file; provider-specific options are passed as keyword arguments."""
        return await self._aupload_file(file, filename=filename, mime_type=mime_type, **kwargs)

    async def _aupload_file(
        self, file: FileInput, *, filename: str | None = None, mime_type: str | None = None, **kwargs: Any
    ) -> FileMetadata:
        message = "Provider does not support file uploads"
        raise NotImplementedError(message)

    @handle_exceptions(file_operation=True)
    async def alist_files(self, *, limit: int | None = None, **kwargs: Any) -> FilePage:
        """Retrieve one page of files; never automatically fetch subsequent pages."""
        return await self._alist_files(limit=limit, **kwargs)

    async def _alist_files(self, *, limit: int | None = None, **kwargs: Any) -> FilePage:
        message = "Provider does not support file listing"
        raise NotImplementedError(message)

    @handle_exceptions(file_operation=True)
    async def aretrieve_file(self, file_id: str, **kwargs: Any) -> FileMetadata:
        """Retrieve metadata without downloading the file contents."""
        return await self._aretrieve_file(file_id, **kwargs)

    async def _aretrieve_file(self, file_id: str, **kwargs: Any) -> FileMetadata:
        message = "Provider does not support file retrieval"
        raise NotImplementedError(message)

    @handle_exceptions(file_operation=True)
    async def adelete_file(self, file_id: str, **kwargs: Any) -> FileDeleted:
        """Delete a file on its originating provider account."""
        return await self._adelete_file(file_id, **kwargs)

    async def _adelete_file(self, file_id: str, **kwargs: Any) -> FileDeleted:
        message = "Provider does not support file deletion"
        raise NotImplementedError(message)

    def upload_file(
        self, file: FileInput, *, filename: str | None = None, mime_type: str | None = None, **kwargs: Any
    ) -> FileMetadata:
        """Run the synchronous counterpart of :meth:`aupload_file`."""
        allow = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(
            self.aupload_file(file, filename=filename, mime_type=mime_type, **kwargs), allow_running_loop=allow
        )

    def list_files(self, *, limit: int | None = None, **kwargs: Any) -> FilePage:
        """Run the synchronous counterpart of :meth:`alist_files`."""
        allow = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(self.alist_files(limit=limit, **kwargs), allow_running_loop=allow)

    def retrieve_file(self, file_id: str, **kwargs: Any) -> FileMetadata:
        """Run the synchronous counterpart of :meth:`aretrieve_file`."""
        allow = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(self.aretrieve_file(file_id, **kwargs), allow_running_loop=allow)

    def delete_file(self, file_id: str, **kwargs: Any) -> FileDeleted:
        """Run the synchronous counterpart of :meth:`adelete_file`."""
        allow = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(self.adelete_file(file_id, **kwargs), allow_running_loop=allow)

    @asynccontextmanager
    async def adownload_file(
        self, file_id: str, *, chunk_size: int = 65536, **kwargs: Any
    ) -> AsyncIterator[AsyncFileDownload]:
        """Open a download on context entry, exposing headers before reading body chunks.

        Only one chunk is requested at a time. Provider failures are handled
        during iteration as well as when opening the download.
        """
        self._validate_chunk_size(chunk_size)

        async with AsyncExitStack() as stack:
            try:
                response = await stack.enter_async_context(
                    self._adownload_file(file_id, chunk_size=chunk_size, **kwargs)
                )
            except Exception as exc:
                _handle_exception(exc, self.PROVIDER_NAME, file_operation=True)
                return  # pragma: no cover

            async def iterate() -> AsyncGenerator[bytes, None]:
                try:
                    async for chunk in response:
                        yield chunk
                except Exception as exc:
                    _handle_exception(exc, self.PROVIDER_NAME, file_operation=True)

            async with aclosing(iterate()) as chunks:
                yield AsyncFileDownload(status_code=response.status_code, headers=response.headers, chunks=chunks)

    @asynccontextmanager
    async def _adownload_file(
        self, file_id: str, *, chunk_size: int, **kwargs: Any
    ) -> AsyncIterator[AsyncFileDownload]:
        message = "Provider does not support file downloads"
        raise NotImplementedError(message)
        yield  # pragma: no cover

    def _validate_chunk_size(self, chunk_size: int) -> None:
        if chunk_size <= 0:
            message = "chunk_size must be positive"
            raise InvalidRequestError(message, provider_name=self.PROVIDER_NAME)

    @contextmanager
    def download_file(self, file_id: str, *, chunk_size: int = 65536, **kwargs: Any) -> Iterator[FileDownload]:
        """Open a download on context entry without prefetching body chunks.

        One task on the runner loop owns the download from start to finish, so the response and any
        contextvar tokens its transport sets are opened, advanced and closed in a single context.
        The handoff to this thread carries one chunk at a time, so the producer never runs ahead.
        """
        allow = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        self._validate_chunk_size(chunk_size)

        async def open_stream() -> AsyncIterator[AsyncFileDownload | bytes]:
            async def stream() -> AsyncGenerator[AsyncFileDownload | bytes, None]:
                async with self.adownload_file(file_id, chunk_size=chunk_size, **kwargs) as response:
                    yield response
                    async for chunk in response:
                        yield chunk

            return stream()

        chunks = _async_source_to_sync_iter(open_stream, allow_running_loop=allow, on_demand=True)
        try:
            # The first handoff carries headers; later reads request body chunks in the same task.
            response = cast("AsyncFileDownload", next(chunks))
            yield FileDownload(
                status_code=response.status_code, headers=response.headers, chunks=cast("Iterator[bytes]", chunks)
            )
        finally:
            chunks.close()
