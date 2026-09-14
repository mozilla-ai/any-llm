from datetime import datetime
from os import PathLike
from typing import BinaryIO, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

FileInput: TypeAlias = str | PathLike[str] | bytes | BinaryIO
FileOperation: TypeAlias = Literal["upload", "list", "retrieve", "download", "delete"]


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
    type: str | None = None


class FilePage(BaseModel):
    """One bounded page, retaining the provider's continuation format."""

    model_config = ConfigDict(extra="allow")

    data: list[FileMetadata]
    next_page: str | None = None
    has_more: bool | None = None
    first_id: str | None = None
    last_id: str | None = None


class FileDeleted(BaseModel):
    """Provider acknowledgement of file deletion."""

    model_config = ConfigDict(extra="allow")

    id: str
    type: str | None = None
    deleted: bool | None = None
