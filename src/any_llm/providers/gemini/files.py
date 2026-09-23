from __future__ import annotations

import math
import mimetypes
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import ExitStack, asynccontextmanager
from io import BytesIO, IOBase
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast
from urllib.parse import urlencode

from google.genai import types
from google.genai.errors import APIError
from typing_extensions import override

from any_llm._files import FilesMixin
from any_llm.exceptions import InvalidRequestError, ProviderError, ProviderFileNotFoundError, UnsupportedParameterError
from any_llm.types.files import AsyncFileDownload, FileDeleted, FileInput, FileMetadata, FileOperation, FilePage

if TYPE_CHECKING:
    from google import genai

PROVIDER_NAME = "gemini"


def reject_unsupported(names: Iterable[str], additional_message: str | None = None) -> None:
    """Reject unsupported options without converting caller mistakes into provider faults."""
    unsupported = sorted(names)
    if unsupported:
        raise UnsupportedParameterError(", ".join(unsupported), PROVIDER_NAME, additional_message)


def _http_status(exc: APIError) -> int | None:
    """Read an HTTP status from an SDK error response.

    google-genai attaches either an httpx response (``status_code``) or an aiohttp
    response (``status``). ``code`` is not consulted: a hand-built ``APIError`` can
    store an HTTP-looking integer there without a response, and that shape is not
    what the SDK raises.
    """
    response = exc.response
    if response is None:
        return None
    for name in ("status_code", "status"):
        status = getattr(response, name, None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status
    return None


def _raise_if_missing_file(exc: APIError) -> None:
    """Map Gemini's unknown-file 403 onto ``ProviderFileNotFoundError``.

    ``files.get`` and ``files.delete`` answer an unknown ID with 403 and a message
    that the file may not exist, instead of 404. A 403 without that phrase stays
    an authentication error. Invalid API keys are 400, so this does not hide a
    credential failure.
    """
    if _http_status(exc) == 403 and "may not exist" in str(exc).lower():
        raise ProviderFileNotFoundError(
            str(exc),
            original_exception=exc,
            provider_name=PROVIDER_NAME,
            status_code=403,
        ) from exc


def validate_file_id(file_id: str) -> str:
    """Reject empty IDs, dot segments, traversal, and URL delimiters; allow a ``files/`` prefix."""
    if (
        not file_id
        or file_id in {".", ".."}
        or any(character in file_id for character in "\\?#%:")
        or any(character.isspace() for character in file_id)
    ):
        message = "A nonempty provider file ID without path separators, URL delimiters, or whitespace is required"
        raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
    rest = file_id.removeprefix("files/") if file_id.startswith("files/") else file_id
    if not rest or rest in {".", ".."} or "/" in rest:
        message = "A nonempty provider file ID without path separators, URL delimiters, or whitespace is required"
        raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
    return file_id if file_id.startswith("files/") else f"files/{file_id}"


def file_http_options(kwargs: dict[str, Any], *, upload: bool = False) -> types.HttpOptions | None:
    """Map shared timeout/header/retry kwargs onto google-genai HttpOptions."""
    if upload:
        kwargs.setdefault("max_retries", 0)
    timeout = kwargs.pop("timeout", None)
    extra_headers = kwargs.pop("extra_headers", None)
    max_retries = kwargs.pop("max_retries", None)
    fields: dict[str, Any] = {}
    if extra_headers is not None:
        if not isinstance(extra_headers, Mapping):
            message = "extra_headers must be a mapping"
            raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
        fields["headers"] = dict(extra_headers)
    if max_retries is not None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            message = "max_retries must be a non-negative integer"
            raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
        fields["retry_options"] = types.HttpRetryOptions(attempts=max_retries + 1)
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            message = "timeout must be a positive number of seconds"
            raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
        # google-genai takes whole milliseconds; truncating would turn a sub-millisecond
        # timeout into 0.
        fields["timeout"] = max(1, math.ceil(timeout * 1000))
    if not fields:
        return None
    return types.HttpOptions(**fields)


def convert_metadata(result: types.File) -> FileMetadata:
    """Map Gemini file fields onto the shared metadata contract, preserving native extras."""
    data = result.model_dump(exclude_unset=True)
    name = data.pop("name", None)
    if not isinstance(name, str) or not name:
        message = "Files response is missing a resource name"
        raise ProviderError(message, provider_name=PROVIDER_NAME)
    filename = data.pop("display_name", None)
    created_at = data.pop("create_time", None)
    expires_at = data.pop("expiration_time", None)
    status = data.pop("state", None)
    download_uri = data.get("download_uri")
    source = data.get("source")
    downloadable: bool | None = None
    if download_uri:
        downloadable = True
    elif source == "UPLOADED":
        downloadable = False
    return FileMetadata.model_validate(
        {
            "id": name,
            "filename": filename,
            "created_at": created_at,
            "expires_at": expires_at,
            "status": status,
            "downloadable": downloadable,
            **data,
        }
    )


def _open_upload_source(
    stack: ExitStack, file: FileInput, filename: str | None, mime_type: str | None
) -> tuple[IOBase, str, str]:
    """Return the handle, display name, and MIME type for an upload.

    Paths are opened here rather than by the SDK, so that only this open is reported as
    an unreadable path; an ``OSError`` from the transport (aiohttp's ``ClientOSError``
    is one) must propagate as a provider failure. Handles this function opens are closed
    by ``stack``; caller-owned handles are left open.
    """
    if isinstance(file, (str, PathLike)):
        path = Path(file)
        try:
            handle = stack.enter_context(path.open("rb"))
        except OSError as exc:
            message = f"Cannot open upload path {str(path)!r}: {exc.strerror or exc}"
            raise InvalidRequestError(message, original_exception=exc, provider_name=PROVIDER_NAME) from exc
        # The SDK requires a MIME type for a handle, so guess it the way it would for a path.
        guessed = mime_type or mimetypes.guess_type(path.name)[0]
        return cast("IOBase", handle), filename or path.name, guessed or "application/octet-stream"
    if isinstance(file, bytes):
        return stack.enter_context(BytesIO(file)), filename or "upload", mime_type or "application/octet-stream"
    return cast("IOBase", file), filename or "upload", mime_type or "application/octet-stream"


async def upload_file(
    client: genai.Client,
    file: FileInput,
    filename: str | None,
    mime_type: str | None,
    purpose: str | None,
    expires_in: int | None,
    kwargs: dict[str, Any],
) -> FileMetadata:
    """Adapt paths, bytes, and binary streams to google-genai's resumable upload."""
    if purpose is not None:
        reject_unsupported(["purpose"])
    if expires_in is not None:
        reject_unsupported(
            ["expires_in"],
            "Gemini files expire after 48 hours and do not accept a configurable expiry.",
        )
    http_options = file_http_options(kwargs, upload=True)
    reject_unsupported(kwargs)
    with ExitStack() as stack:
        handle, display_name, resolved_mime_type = _open_upload_source(stack, file, filename, mime_type)
        config = types.UploadFileConfig(
            mime_type=resolved_mime_type,
            display_name=display_name,
            http_options=http_options,
        )
        result = await client.aio.files.upload(file=handle, config=config)
    return convert_metadata(result)


async def list_files(
    client: genai.Client,
    limit: int | None,
    cursor: str | None,
    purpose: str | None,
    kwargs: dict[str, Any],
) -> FilePage:
    """Fetch one SDK page via the pager's current page, never iterating subsequent pages."""
    if purpose is not None:
        reject_unsupported(["purpose"])
    http_options = file_http_options(kwargs)
    reject_unsupported(kwargs)
    page_size: int | None = None
    page_token: str | None = None
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            message = "limit must be a positive integer"
            raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
        page_size = limit
    if cursor is not None:
        if not isinstance(cursor, str) or not cursor:
            message = "cursor must be a nonempty string"
            raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
        page_token = cursor
    config = None
    if http_options is not None or page_size is not None or page_token is not None:
        config = types.ListFilesConfig(http_options=http_options, page_size=page_size, page_token=page_token)
    pager = await client.aio.files.list(config=config)
    next_cursor = pager.config.get("page_token") or None
    return FilePage(data=[convert_metadata(item) for item in pager.page], next_cursor=next_cursor)


@asynccontextmanager
async def stream_download(
    client: genai.Client,
    file_id: str,
    *,
    chunk_size: int,
    http_options: types.HttpOptions | None,
) -> AsyncIterator[AsyncFileDownload]:
    """Stream a generated-file download without buffering the body first.

    google-genai's ``files.download()`` either returns the full bytes or writes
    them to a destination; it does not expose status, headers, and a lazy
    iterator. The Files API contract needs all three, so this opens the same
    ``files/{id}:download?alt=media`` request through the SDK HTTP client and
    yields chunks from the live response.
    """
    name = file_id.removeprefix("files/")
    path = f"files/{name}:download?{urlencode({'alt': 'media'})}"
    api_client = client._api_client
    http_request = api_client._build_request("get", path, {}, http_options)
    httpx_client = api_client._async_httpx_client
    if httpx_client is None:
        message = "Gemini file downloads require the google-genai async HTTP client"
        raise ProviderError(message, provider_name=PROVIDER_NAME)
    request = httpx_client.build_request(
        method=http_request.method,
        url=http_request.url,
        headers=http_request.headers,
        timeout=http_request.timeout,
    )
    response = await httpx_client.send(request, stream=True)  # type: ignore[arg-type]
    try:
        try:
            await APIError.raise_for_async_response(response)
        except APIError as exc:
            _raise_if_missing_file(exc)
            raise
        yield AsyncFileDownload(
            status_code=response.status_code,
            headers=response.headers.copy(),
            chunks=response.aiter_bytes(chunk_size=chunk_size),
        )
    finally:
        await response.aclose()


class GeminiFileMethods(FilesMixin):
    """Opt-in Files implementation for the Gemini Developer API.

    Vertex AI shares ``GoogleProvider`` without this mixin and does not get Files.
    """

    SUPPORTED_FILE_OPERATIONS: ClassVar[frozenset[FileOperation]]
    client: genai.Client

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
        file_id = validate_file_id(file_id)
        http_options = file_http_options(kwargs)
        reject_unsupported(kwargs)
        config = types.GetFileConfig(http_options=http_options) if http_options is not None else None
        try:
            result = await self.client.aio.files.get(name=file_id, config=config)
        except APIError as exc:
            _raise_if_missing_file(exc)
            raise
        return convert_metadata(result)

    @override
    async def _adelete_file(self, file_id: str, **kwargs: Any) -> FileDeleted:
        file_id = validate_file_id(file_id)
        http_options = file_http_options(kwargs)
        reject_unsupported(kwargs)
        config = types.DeleteFileConfig(http_options=http_options) if http_options is not None else None
        try:
            await self.client.aio.files.delete(name=file_id, config=config)
        except APIError as exc:
            _raise_if_missing_file(exc)
            raise
        return FileDeleted(id=file_id)

    @override
    @asynccontextmanager
    async def _adownload_file(
        self, file_id: str, *, chunk_size: int, **kwargs: Any
    ) -> AsyncIterator[AsyncFileDownload]:
        file_id = validate_file_id(file_id)
        http_options = file_http_options(kwargs)
        reject_unsupported(kwargs)
        get_config = types.GetFileConfig(http_options=http_options) if http_options is not None else None
        try:
            metadata = await self.client.aio.files.get(name=file_id, config=get_config)
        except APIError as exc:
            _raise_if_missing_file(exc)
            raise
        if not metadata.download_uri:
            message = "Gemini user-uploaded files cannot be downloaded; only generated files with a download_uri can"
            raise InvalidRequestError(message, provider_name=self.PROVIDER_NAME)
        async with stream_download(self.client, file_id, chunk_size=chunk_size, http_options=http_options) as download:
            yield download
