from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from google.genai import types
from google.genai.errors import APIError
from typing_extensions import override

from any_llm._caches import CachesMixin
from any_llm.exceptions import InvalidRequestError, ProviderCacheNotFoundError, ProviderError
from any_llm.types.caches import CachedContent, CacheDeleted, CacheOperation, CachePage, CacheUsage

from .files import _http_status, file_http_options, reject_unsupported
from .utils import _convert_messages, _convert_tool_choice, _convert_tool_spec

if TYPE_CHECKING:
    from datetime import datetime

    from google import genai

_COLLECTION = "cachedContents"
_INVALID_CACHE_ID = (
    "A nonempty cache ID is required: a bare ID, cachedContents/{id}, "
    "or projects/{project}/locations/{location}/cachedContents/{id}"
)


def _is_segment(value: str) -> bool:
    return bool(value) and value not in {".", ".."}


def validate_cache_id(cache_id: str, provider_name: str) -> str:
    """Canonicalize a cache ID to a resource name, rejecting traversal and URL delimiters.

    Vertex AI names caches ``projects/{project}/locations/{location}/cachedContents/{id}``,
    so that full form is accepted beside ``cachedContents/{id}`` and the bare ID.
    """
    if (
        not isinstance(cache_id, str)
        or any(character in cache_id for character in "\\?#%:")
        or any(character.isspace() for character in cache_id)
    ):
        raise InvalidRequestError(_INVALID_CACHE_ID, provider_name=provider_name)
    segments = cache_id.split("/")
    if len(segments) == 1:
        segments = [_COLLECTION, *segments]
    valid = (len(segments) == 2 and segments[0] == _COLLECTION) or (
        len(segments) == 6
        and segments[0] == "projects"
        and segments[2] == "locations"
        and segments[4] == _COLLECTION
        and _is_segment(segments[1])
        and _is_segment(segments[3])
    )
    if not valid or not _is_segment(segments[-1]):
        raise InvalidRequestError(_INVALID_CACHE_ID, provider_name=provider_name)
    return "/".join(segments)


def _raise_if_missing_cache(exc: APIError, provider_name: str) -> None:
    """Map Gemini's unknown-cache 403 onto ``ProviderCacheNotFoundError``.

    The Developer API answers an unknown or expired cache with 403 "CachedContent not
    found (or permission denied)" instead of 404. A 403 without "not found" stays an
    authentication error.
    """
    if _http_status(exc) == 403 and "not found" in str(exc).lower():
        raise ProviderCacheNotFoundError(
            str(exc), original_exception=exc, provider_name=provider_name, status_code=403
        ) from exc


def convert_cache(result: types.CachedContent, provider_name: str) -> CachedContent:
    """Map google-genai cache fields onto the shared contract, preserving native extras."""
    data = result.model_dump(exclude_unset=True, exclude_none=True)
    name = data.pop("name", None)
    if not isinstance(name, str) or not name:
        message = "Cache response is missing a resource name"
        raise ProviderError(message, provider_name=provider_name)
    usage: CacheUsage | None = None
    if (usage_metadata := data.pop("usage_metadata", None)) is not None:
        total_tokens = usage_metadata.pop("total_token_count", None)
        usage = CacheUsage.model_validate({"total_tokens": total_tokens, **usage_metadata})
    return CachedContent.model_validate(
        {
            "id": name,
            "model": data.pop("model", None),
            "display_name": data.pop("display_name", None),
            "created_at": data.pop("create_time", None),
            "updated_at": data.pop("update_time", None),
            "expires_at": data.pop("expire_time", None),
            "usage": usage,
            **data,
        }
    )


def _expiry_fields(ttl: int | None, expires_at: datetime | None) -> dict[str, Any]:
    if ttl is not None:
        return {"ttl": f"{ttl}s"}
    if expires_at is not None:
        return {"expire_time": expires_at}
    return {}


async def create_cache(
    client: genai.Client,
    provider_name: str,
    model: str,
    messages: list[dict[str, Any]] | None,
    tools: list[dict[str, Any] | Any] | None,
    tool_choice: str | dict[str, Any] | None,
    ttl: int | None,
    expires_at: datetime | None,
    display_name: str | None,
    kwargs: dict[str, Any],
) -> CachedContent:
    """Build the cache from the same conversions a completion applies to these inputs."""
    http_options = file_http_options(kwargs, provider_name=provider_name)
    reject_unsupported(kwargs, provider_name=provider_name)
    fields: dict[str, Any] = {"http_options": http_options, "display_name": display_name}
    fields.update(_expiry_fields(ttl, expires_at))
    if messages:
        contents, system_instruction = _convert_messages(messages, provider_name=provider_name)
        if contents:
            fields["contents"] = contents
        if system_instruction:
            fields["system_instruction"] = system_instruction
    if tools:
        fields["tools"] = _convert_tool_spec(tools, provider_name)
    if tool_choice is not None:
        fields["tool_config"] = _convert_tool_choice(tool_choice, provider_name)
    config = types.CreateCachedContentConfig(**fields)
    return convert_cache(await client.aio.caches.create(model=model, config=config), provider_name)


async def list_caches(
    client: genai.Client, provider_name: str, limit: int | None, cursor: str | None, kwargs: dict[str, Any]
) -> CachePage:
    """Fetch one SDK page via the pager's current page, never iterating subsequent pages."""
    http_options = file_http_options(kwargs, provider_name=provider_name)
    reject_unsupported(kwargs, provider_name=provider_name)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        message = "limit must be a positive integer"
        raise InvalidRequestError(message, provider_name=provider_name)
    if cursor is not None and (not isinstance(cursor, str) or not cursor):
        message = "cursor must be a nonempty string"
        raise InvalidRequestError(message, provider_name=provider_name)
    config = None
    if http_options is not None or limit is not None or cursor is not None:
        config = types.ListCachedContentsConfig(http_options=http_options, page_size=limit, page_token=cursor)
    pager = await client.aio.caches.list(config=config)
    next_cursor = pager.config.get("page_token") or None
    return CachePage(data=[convert_cache(item, provider_name) for item in pager.page], next_cursor=next_cursor)


class GeminiCacheMethods(CachesMixin):
    """Context caching through google-genai's ``client.aio.caches``.

    The Developer API and Vertex AI expose the same caches surface, so both providers use this mixin.
    """

    SUPPORTED_CACHE_OPERATIONS: ClassVar[frozenset[CacheOperation]] = frozenset(
        {"create", "list", "retrieve", "update", "delete"}
    )
    client: genai.Client

    @override
    async def _acreate_cache(
        self,
        model: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any] | Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        ttl: int | None = None,
        expires_at: datetime | None = None,
        display_name: str | None = None,
        **kwargs: Any,
    ) -> CachedContent:
        return await create_cache(
            self.client,
            self.PROVIDER_NAME,
            model,
            messages,
            tools,
            tool_choice,
            ttl,
            expires_at,
            display_name,
            kwargs,
        )

    @override
    async def _alist_caches(self, *, limit: int | None = None, cursor: str | None = None, **kwargs: Any) -> CachePage:
        return await list_caches(self.client, self.PROVIDER_NAME, limit, cursor, kwargs)

    @override
    async def _aretrieve_cache(self, cache_id: str, **kwargs: Any) -> CachedContent:
        name = validate_cache_id(cache_id, self.PROVIDER_NAME)
        http_options = file_http_options(kwargs, provider_name=self.PROVIDER_NAME)
        reject_unsupported(kwargs, provider_name=self.PROVIDER_NAME)
        config = types.GetCachedContentConfig(http_options=http_options) if http_options is not None else None
        try:
            result = await self.client.aio.caches.get(name=name, config=config)
        except APIError as exc:
            _raise_if_missing_cache(exc, self.PROVIDER_NAME)
            raise
        return convert_cache(result, self.PROVIDER_NAME)

    @override
    async def _aupdate_cache(
        self, cache_id: str, *, ttl: int | None = None, expires_at: datetime | None = None, **kwargs: Any
    ) -> CachedContent:
        name = validate_cache_id(cache_id, self.PROVIDER_NAME)
        http_options = file_http_options(kwargs, provider_name=self.PROVIDER_NAME)
        reject_unsupported(kwargs, provider_name=self.PROVIDER_NAME)
        config = types.UpdateCachedContentConfig(http_options=http_options, **_expiry_fields(ttl, expires_at))
        try:
            result = await self.client.aio.caches.update(name=name, config=config)
        except APIError as exc:
            _raise_if_missing_cache(exc, self.PROVIDER_NAME)
            raise
        return convert_cache(result, self.PROVIDER_NAME)

    @override
    async def _adelete_cache(self, cache_id: str, **kwargs: Any) -> CacheDeleted:
        name = validate_cache_id(cache_id, self.PROVIDER_NAME)
        http_options = file_http_options(kwargs, provider_name=self.PROVIDER_NAME)
        reject_unsupported(kwargs, provider_name=self.PROVIDER_NAME)
        config = types.DeleteCachedContentConfig(http_options=http_options) if http_options is not None else None
        try:
            await self.client.aio.caches.delete(name=name, config=config)
        except APIError as exc:
            _raise_if_missing_cache(exc, self.PROVIDER_NAME)
            raise
        return CacheDeleted(id=name)
