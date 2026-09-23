from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

CacheOperation: TypeAlias = Literal["create", "list", "retrieve", "update", "delete"]


class CacheUsage(BaseModel):
    """Token accounting for cached content; provider-specific counts are preserved as extras."""

    model_config = ConfigDict(extra="allow")

    total_tokens: int | None = None


class CachedContent(BaseModel):
    """Provider-hosted context cache metadata; unavailable fields stay unknown.

    IDs belong to the originating provider account and are the provider's resource
    name. Provider-specific fields are preserved in ``model_extra`` and ``model_dump()``.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    model: str | None = None
    display_name: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    expires_at: datetime | None = None
    usage: CacheUsage | None = None


class CachePage(BaseModel):
    """One bounded page with an opaque continuation cursor; None means the final page."""

    model_config = ConfigDict(extra="allow")

    data: list[CachedContent]
    next_cursor: str | None = None


class CacheDeleted(BaseModel):
    """Provider acknowledgement of cache deletion."""

    model_config = ConfigDict(extra="allow")

    id: str
    deleted: bool | None = None
