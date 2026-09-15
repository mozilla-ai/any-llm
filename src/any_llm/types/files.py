from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from os import PathLike
from typing import BinaryIO, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict
from typing_extensions import override

FileInput: TypeAlias = str | PathLike[str] | bytes | BinaryIO
FileOperation: TypeAlias = Literal["upload", "list", "retrieve", "download", "delete"]


@dataclass(frozen=True)
class AsyncFileDownload(AsyncIterator[bytes]):
    """An open download with response headers and a single-use async byte iterator.

    Consume only inside the download context, which owns response cleanup.
    Headers describe the upstream response; body chunks may be decoded.
    """

    status_code: int
    headers: Mapping[str, str] = field(repr=False)
    chunks: AsyncIterator[bytes] = field(repr=False)

    @override
    def __aiter__(self) -> Self:
        return self

    @override
    async def __anext__(self) -> bytes:
        return await anext(self.chunks)


@dataclass(frozen=True)
class FileDownload(Iterator[bytes]):
    """An open download with response headers and a single-use sync byte iterator.

    Consume only inside the download context, which owns response cleanup.
    Headers describe the upstream response; body chunks may be decoded.
    """

    status_code: int
    headers: Mapping[str, str] = field(repr=False)
    chunks: Iterator[bytes] = field(repr=False)

    @override
    def __iter__(self) -> Self:
        return self

    @override
    def __next__(self) -> bytes:
        return next(self.chunks)


class FileMetadata(BaseModel):
    """Provider-hosted file metadata; unavailable fields stay unknown.

    IDs belong to the originating provider account. Provider-specific fields
    are preserved in ``model_extra`` and ``model_dump()``.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    filename: str | None = None
    size_bytes: int | None = None
    mime_type: str | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None
    downloadable: bool | None = None
    purpose: str | None = None
    status: str | None = None


class FilePage(BaseModel):
    """One bounded page with an opaque continuation cursor; None means the final page."""

    model_config = ConfigDict(extra="allow")

    data: list[FileMetadata]
    next_cursor: str | None = None


class FileDeleted(BaseModel):
    """Provider acknowledgement of file deletion."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: str | None = None
    deleted: bool | None = None
