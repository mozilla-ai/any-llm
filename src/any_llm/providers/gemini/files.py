from __future__ import annotations

import mimetypes
import re
from contextlib import ExitStack
from io import BytesIO, IOBase
from os import PathLike
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

from google.genai import errors, types

from any_llm.exceptions import (
    InvalidRequestError,
    ProviderError,
    ProviderFileNotFoundError,
    UnsupportedParameterError,
)
from any_llm.types.files import FileDeleted, FileInput, FileMetadata, FilePage
from any_llm.utils.exception_handler import convert_exception, unified_exceptions_enabled

if TYPE_CHECKING:
    from collections.abc import Iterable

    from google import genai

PROVIDER_NAME = "gemini"
_DEFAULT_MIME_TYPE = "application/octet-stream"
# Gemini file resources are named "files/<id>", and the SDK accepts either spelling.
_FILE_ID_PATTERN = re.compile(r"^(?:files/)?[A-Za-z0-9_-]+$")
# Gemini will not say whether a file exists, so a file-scoped call for an unknown ID comes
# back as 403 PERMISSION_DENIED ("You do not have permission to access the File X or it may
# not exist.") instead of 404. An invalid API key is reported as 400, so a 403 naming a
# possibly absent file is never a credential failure.
_MISSING_FILE_MESSAGE = re.compile(r"may not exist|does not exist|not found", re.IGNORECASE)
# Fields that become shared FileMetadata attributes; everything else stays in extras.
_MAPPED_FIELDS = frozenset(
    {"name", "display_name", "size_bytes", "mime_type", "create_time", "expiration_time", "state"}
)


def reject_unsupported(names: Iterable[str], additional_message: str | None = None) -> None:
    """Reject unsupported options without converting caller mistakes into provider faults."""
    unsupported = sorted(names)
    if unsupported:
        raise UnsupportedParameterError(", ".join(unsupported), PROVIDER_NAME, additional_message)


def validate_file_id(file_id: str) -> None:
    """Reject IDs outside Gemini's "files/<id>" resource naming."""
    if not _FILE_ID_PATTERN.match(file_id):
        message = "A Gemini file ID of the form 'files/<id>' or '<id>' is required"
        raise InvalidRequestError(message, provider_name=PROVIDER_NAME)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        message = f"{name} must be a positive integer"
        raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
    return value


def request_http_options(kwargs: dict[str, Any], *, upload: bool = False) -> types.HttpOptions | None:
    """Translate the shared request options into a per-request ``HttpOptions``.

    Values left unset here fall through to the options the client was built with.
    """
    http_options: dict[str, Any] = {}

    timeout = kwargs.pop("timeout", None)
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            message = "timeout must be a positive number of seconds"
            raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
        http_options["timeout"] = int(timeout * 1000)

    if upload and "extra_headers" in kwargs:
        # The SDK replaces the request headers when it prepares the resumable upload session,
        # so accepting them here would silently drop them.
        reject_unsupported(["extra_headers"], "Gemini uploads do not accept extra_headers.")
    extra_headers = kwargs.pop("extra_headers", None)
    if extra_headers is not None:
        http_options["headers"] = dict(extra_headers)

    max_retries = kwargs.pop("max_retries", 0 if upload else None)
    if max_retries is not None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            message = "max_retries must be a nonnegative integer"
            raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
        # attempts counts the original request, so one attempt means no retries. This governs
        # the SDK's request wrapper only: the resumable upload's byte transfer runs its own
        # MAX_RETRY_COUNT loop in _async_upload_fd, which no HttpRetryOptions reaches.
        http_options["retry_options"] = types.HttpRetryOptions(attempts=max_retries + 1)

    return types.HttpOptions(**http_options) if http_options else None


def convert_metadata(result: types.File) -> FileMetadata:
    """Convert a Gemini file resource, preserving native fields and absent values."""
    if result.name is None:
        message = "Gemini file response is missing its resource name"
        raise ProviderError(message, provider_name=PROVIDER_NAME)
    native = result.model_dump(exclude_unset=True, mode="json")
    extras = {key: value for key, value in native.items() if key not in _MAPPED_FIELDS}
    return FileMetadata(
        id=result.name,
        filename=result.display_name,
        size_bytes=result.size_bytes,
        mime_type=result.mime_type,
        created_at=result.create_time,
        expires_at=result.expiration_time,
        # Only uploads are known to be undownloadable; other sources stay unknown because
        # any-llm does not expose Gemini downloads.
        downloadable=False if result.source == types.FileSource.UPLOADED else None,
        status=result.state.value if result.state is not None else None,
        **extras,
    )


async def upload_file(
    client: genai.Client,
    file: FileInput,
    filename: str | None,
    mime_type: str | None,
    purpose: str | None,
    expires_in: int | None,
    kwargs: dict[str, Any],
) -> FileMetadata:
    """Map normalized upload options to the Gemini Files resource."""
    if purpose is not None:
        reject_unsupported(["purpose"], "Gemini files are not classified by purpose.")
    if expires_in is not None:
        reject_unsupported(["expires_in"], "Gemini files expire 48 hours after upload.")
    http_options = request_http_options(kwargs, upload=True)
    reject_unsupported(kwargs)

    with ExitStack() as stack:
        content: IO[bytes]
        if isinstance(file, (str, PathLike)):
            path = Path(file)
            try:
                content = stack.enter_context(path.open("rb"))  # noqa: ASYNC230 (the SDK reads upload handles synchronously)
            except OSError as exc:
                # Without this, exception conversion classifies FileNotFoundError by its type name
                # as ModelNotFoundError.
                message = f"Cannot open upload path {str(path)!r}: {exc.strerror or exc}"
                raise InvalidRequestError(message, original_exception=exc, provider_name=PROVIDER_NAME) from exc
            filename = filename or path.name
            mime_type = mime_type or mimetypes.guess_type(path.name)[0]
        elif isinstance(file, bytes):
            content = BytesIO(file)
        else:
            content = file
        config = types.UploadFileConfig(
            http_options=http_options,
            # The SDK refuses to guess a MIME type for a handle, so a default is always supplied.
            mime_type=mime_type or _DEFAULT_MIME_TYPE,
            display_name=filename,
        )
        # Every input reaches the SDK as a binary stream, which its IOBase branch accepts;
        # the annotation on FileInput is the wider typing.BinaryIO.
        result = await client.aio.files.upload(file=cast("IOBase", content), config=config)
    return convert_metadata(result)


async def list_files(
    client: genai.Client,
    limit: int | None,
    cursor: str | None,
    purpose: str | None,
    kwargs: dict[str, Any],
) -> FilePage:
    """Fetch exactly one page and translate its continuation to the common cursor."""
    if purpose is not None:
        reject_unsupported(["purpose"], "Gemini files are not classified by purpose.")
    config = types.ListFilesConfig(http_options=request_http_options(kwargs))
    reject_unsupported(kwargs)
    if limit is not None:
        config.page_size = _positive_int(limit, "limit")
    if cursor is not None:
        if not cursor.strip():
            message = "cursor must be a nonempty page token"
            raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
        config.page_token = cursor

    pager = await client.aio.files.list(config=config)
    # The pager holds the token for the page after the one already fetched; reading it here
    # keeps listing to a single request instead of walking the remaining pages.
    next_cursor = pager.config.get("page_token") or None
    return FilePage(data=[convert_metadata(item) for item in pager.page], next_cursor=next_cursor)


def _reject_missing_file(exc: errors.APIError, unified_exceptions: bool | None) -> None:
    """Report an unusable file ID as a missing file rather than a permission failure.

    Only the unified-exception path is translated; otherwise the SDK exception is left to
    propagate untouched, as it does for every other provider failure.
    """
    if exc.code != 403 or not _MISSING_FILE_MESSAGE.search(exc.message or ""):
        return
    if not unified_exceptions_enabled(unified_exceptions):
        return
    converted = convert_exception(exc, PROVIDER_NAME)
    raise ProviderFileNotFoundError(
        message=converted.message,
        original_exception=exc,
        provider_name=PROVIDER_NAME,
        status_code=exc.code,
        code=converted.code,
        param=converted.param,
        error_type=converted.error_type,
    ) from exc


async def retrieve_file(
    client: genai.Client, file_id: str, kwargs: dict[str, Any], unified_exceptions: bool | None = None
) -> FileMetadata:
    """Retrieve metadata, including the processing state, without downloading contents."""
    config = types.GetFileConfig(http_options=request_http_options(kwargs))
    reject_unsupported(kwargs)
    validate_file_id(file_id)
    try:
        result = await client.aio.files.get(name=file_id, config=config)
    except errors.APIError as exc:
        _reject_missing_file(exc, unified_exceptions)
        raise
    return convert_metadata(result)


async def delete_file(
    client: genai.Client, file_id: str, kwargs: dict[str, Any], unified_exceptions: bool | None = None
) -> FileDeleted:
    """Delete a file, reporting deletion from the acknowledged call rather than a response body."""
    config = types.DeleteFileConfig(http_options=request_http_options(kwargs))
    reject_unsupported(kwargs)
    validate_file_id(file_id)
    try:
        await client.aio.files.delete(name=file_id, config=config)
    except errors.APIError as exc:
        _reject_missing_file(exc, unified_exceptions)
        raise
    # Gemini acknowledges deletion with an empty body, so a returned call is the only signal.
    return FileDeleted(id=file_id, deleted=True)
