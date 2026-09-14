from __future__ import annotations

from contextlib import ExitStack
from os import PathLike
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any
from urllib.parse import quote

from any_llm.exceptions import InvalidRequestError, UnsupportedParameterError
from any_llm.types.files import FileInput, FileMetadata, FilePage

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anthropic import AsyncAnthropic
    from anthropic._types import RequestOptions

PROVIDER_NAME = "anthropic"


def reject_unsupported(names: Iterable[str], additional_message: str | None = None) -> None:
    """Reject caller options the Files API does not accept.

    Raises an ``AnyLLMError`` subclass so that a caller mistake is not converted into a
    provider fault by the unified-exception handler.
    """
    unsupported = sorted(names)
    if unsupported:
        raise UnsupportedParameterError(", ".join(unsupported), PROVIDER_NAME, additional_message)


def file_path(file_id: str) -> str:
    if not file_id or file_id in {".", ".."}:
        message = "A nonempty provider file ID is required"
        raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
    return f"/v1/files/{quote(file_id, safe='')}"


def request_options(client: AsyncAnthropic, kwargs: dict[str, Any]) -> tuple[AsyncAnthropic, RequestOptions]:
    options: RequestOptions = {}
    headers = {name.lower(): value for name, value in (kwargs.pop("extra_headers", {}) or {}).items()}
    betas = kwargs.pop("betas", None)
    if betas is not None:
        headers["anthropic-beta"] = ",".join(betas)
    options["headers"] = headers
    if "timeout" in kwargs:
        options["timeout"] = kwargs.pop("timeout")
    if "max_retries" in kwargs:
        client = client.with_options(max_retries=kwargs.pop("max_retries"))
    return client, options


async def upload_file(
    client: AsyncAnthropic, file: FileInput, filename: str | None, mime_type: str | None, kwargs: dict[str, Any]
) -> FileMetadata:
    kwargs.setdefault("max_retries", 0)
    client, options = request_options(client, kwargs)
    reject_unsupported(
        set(kwargs) - {"expires_in_seconds"},
        "Anthropic Files upload accepts only expires_in_seconds.",
    )
    if "expires_in_seconds" in kwargs:
        expiry = kwargs["expires_in_seconds"]
        if not isinstance(expiry, int) or isinstance(expiry, bool) or not 3600 <= expiry <= 7776000:
            message = "expires_in_seconds must be an integer between 3600 and 7776000"
            raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
    content: bytes | IO[bytes]
    with ExitStack() as stack:
        if isinstance(file, (str, PathLike)):
            path = Path(file)
            content = stack.enter_context(path.open("rb"))  # noqa: ASYNC230  (the SDK multipart encoder requires a synchronous binary handle)
            filename = filename or path.name
        else:
            content = file
        filename = filename or "upload"
        options["headers"] = {**(options.get("headers") or {}), "Content-Type": "multipart/form-data"}
        result = await client.post(
            "/v1/files",
            cast_to=dict[str, Any],
            body=kwargs,
            files=[("file", (filename, content, mime_type or "application/octet-stream"))],
            options=options,
        )
    return FileMetadata.model_validate(result)


async def list_files(client: AsyncAnthropic, limit: int | None, kwargs: dict[str, Any]) -> FilePage:
    client, options = request_options(client, kwargs)
    if limit is not None:
        if not 1 <= limit <= 1000:
            message = "limit must be between 1 and 1000"
            raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
        kwargs["limit"] = limit
    effective_headers = {name.lower(): value for name, value in client.default_headers.items()}
    effective_headers.update(options.get("headers") or {})
    beta_header = effective_headers.get("anthropic-beta", "")
    betas_in_effect = {value.strip() for value in beta_header.split(",")} if isinstance(beta_header, str) else set()
    legacy = "files-api-2025-04-14" in betas_in_effect
    allowed = {"before_id", "after_id", "limit", "order"} if legacy else {"page", "ids", "limit"}
    reject_unsupported(
        set(kwargs) - allowed,
        "Do not mix current (page, ids) and legacy (before_id, after_id, order) pagination arguments.",
    )
    if "ids" in kwargs:
        ids = list(dict.fromkeys(kwargs["ids"]))
        if "page" in kwargs or "limit" in kwargs or not 1 <= len(ids) <= 100:
            message = "ids requires 1 to 100 IDs and cannot be combined with page or limit"
            raise InvalidRequestError(message, provider_name=PROVIDER_NAME)
        kwargs["ids"] = ids
    options["params"] = kwargs
    result = await client.get("/v1/files", cast_to=dict[str, Any], options=options)
    return FilePage.model_validate(result)
