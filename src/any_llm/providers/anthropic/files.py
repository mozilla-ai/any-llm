from __future__ import annotations

from contextlib import ExitStack
from os import PathLike
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

from any_llm.exceptions import InvalidRequestError, UnsupportedParameterError
from any_llm.types.files import FileInput, FileMetadata, FilePage

from .base import _pop_anthropic_beta_header

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anthropic import AsyncAnthropic
    from anthropic.types import FileMetadata as AnthropicFileMetadata

PROVIDER_NAME = "anthropic"
_LEGACY_BETA = "files-api-2025-04-14"


def reject_unsupported(names: Iterable[str], additional_message: str | None = None) -> None:
    """Reject unsupported options without converting caller mistakes into provider faults."""
    unsupported = sorted(names)
    if unsupported:
        raise UnsupportedParameterError(", ".join(unsupported), PROVIDER_NAME, additional_message)


def validate_file_id(file_id: str) -> None:
    """Reject IDs that are empty, dot segments, or carry path separators or whitespace."""
    if not file_id or file_id in {".", ".."} or "/" in file_id or "\\" in file_id or file_id != file_id.strip():
        message = "A nonempty provider file ID without path separators is required"
        raise InvalidRequestError(message, provider_name=PROVIDER_NAME)


def request_options(client: AsyncAnthropic, kwargs: dict[str, Any]) -> tuple[AsyncAnthropic, dict[str, Any]]:
    """Extract SDK options and merge beta headers without mutating caller headers."""
    defaults: dict[str, Any] = {"extra_headers": client.default_headers}
    betas = list(
        dict.fromkeys(
            [
                *_pop_anthropic_beta_header(defaults),
                *_pop_anthropic_beta_header(kwargs),
                *(kwargs.pop("betas", None) or []),
            ]
        )
    )
    if _LEGACY_BETA in betas:
        reject_unsupported(["betas"], "Legacy Files pagination is not supported.")
    headers = dict(kwargs.pop("extra_headers", None) or {})
    if betas:
        headers["anthropic-beta"] = ",".join(betas)
    options: dict[str, Any] = {"extra_headers": headers}
    if "timeout" in kwargs:
        options["timeout"] = kwargs.pop("timeout")
    if "max_retries" in kwargs:
        client = client.with_options(max_retries=kwargs.pop("max_retries"))
    return client, options


def convert_metadata(result: AnthropicFileMetadata) -> FileMetadata:
    """Convert Anthropic metadata, preserving extras and absent values."""
    return FileMetadata.model_validate(result.model_dump(exclude_unset=True))


async def upload_file(
    client: AsyncAnthropic,
    file: FileInput,
    filename: str | None,
    mime_type: str | None,
    purpose: str | None,
    expires_in: int | None,
    kwargs: dict[str, Any],
) -> FileMetadata:
    """Map normalized upload options to Anthropic's SDK Files resource."""
    if purpose is not None:
        reject_unsupported(["purpose"])
    kwargs.setdefault("max_retries", 0)
    client, options = request_options(client, kwargs)
    reject_unsupported(kwargs)
    if expires_in is not None:
        if isinstance(expires_in, bool) or not isinstance(expires_in, int) or expires_in <= 0:
            message = "expires_in must be a positive integer number of seconds"
            raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
        options["expires_in_seconds"] = expires_in
    content: bytes | IO[bytes]
    with ExitStack() as stack:
        if isinstance(file, (str, PathLike)):
            path = Path(file)
            try:
                content = stack.enter_context(path.open("rb"))  # noqa: ASYNC230 (SDK multipart encoding uses synchronous handles)
            except OSError as exc:
                # Without this, exception conversion classifies FileNotFoundError by its type name as ModelNotFoundError.
                message = f"Cannot open upload path {str(path)!r}: {exc.strerror or exc}"
                raise InvalidRequestError(message, original_exception=exc, provider_name=PROVIDER_NAME) from exc
            filename = filename or path.name
        else:
            content = file
        result = await client.files.upload(
            file=(filename or "upload", content, mime_type or "application/octet-stream"), **options
        )
    return convert_metadata(result)


async def list_files(
    client: AsyncAnthropic,
    limit: int | None,
    cursor: str | None,
    purpose: str | None,
    kwargs: dict[str, Any],
) -> FilePage:
    """Fetch one SDK page and translate its continuation to the common cursor."""
    if purpose is not None:
        reject_unsupported(["purpose"])
    client, options = request_options(client, kwargs)
    reject_unsupported(set(kwargs) - {"ids"})
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            message = "limit must be a positive integer"
            raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
        options["limit"] = limit
    if cursor is not None:
        options["page"] = cursor
    result = await client.files.list(**kwargs, **options)
    native = result.model_dump(exclude_unset=True)
    extras = {key: value for key, value in native.items() if key not in {"data", "next_page", "next_cursor"}}
    return FilePage(data=[convert_metadata(item) for item in result.data], next_cursor=result.next_page, **extras)
