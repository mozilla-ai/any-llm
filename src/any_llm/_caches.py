from collections.abc import Callable
from datetime import datetime
from typing import Any, ClassVar

from any_llm.constants import INSIDE_NOTEBOOK
from any_llm.exceptions import InvalidRequestError
from any_llm.tools import prepare_tools
from any_llm.types.caches import CachedContent, CacheDeleted, CacheOperation, CachePage
from any_llm.types.completion import ChatCompletionMessage
from any_llm.utils.aio import run_async_in_sync
from any_llm.utils.exception_handler import handle_exceptions


class CachesMixin:
    """Public provider context-cache operations and unsupported-provider defaults."""

    PROVIDER_NAME: str
    BUILT_IN_TOOLS: ClassVar[list[Any] | None] = None

    SUPPORTED_CACHE_OPERATIONS: ClassVar[frozenset[CacheOperation]] = frozenset()

    def _validate_expiry(self, ttl: int | None, expires_at: datetime | None, *, required: bool) -> None:
        if ttl is not None and expires_at is not None:
            message = "ttl and expires_at are mutually exclusive"
            raise InvalidRequestError(message, provider_name=self.PROVIDER_NAME)
        if required and ttl is None and expires_at is None:
            message = "Exactly one of ttl or expires_at is required"
            raise InvalidRequestError(message, provider_name=self.PROVIDER_NAME)
        if ttl is not None and (isinstance(ttl, bool) or not isinstance(ttl, int) or ttl <= 0):
            message = "ttl must be a positive integer number of seconds"
            raise InvalidRequestError(message, provider_name=self.PROVIDER_NAME)
        if expires_at is not None and (not isinstance(expires_at, datetime) or expires_at.utcoffset() is None):
            message = "expires_at must be a timezone-aware datetime"
            raise InvalidRequestError(message, provider_name=self.PROVIDER_NAME)

    @handle_exceptions()
    async def acreate_cache(
        self,
        model: str,
        *,
        messages: list[dict[str, Any] | ChatCompletionMessage] | None = None,
        tools: list[dict[str, Any] | Callable[..., Any]] | Any | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        ttl: int | None = None,
        expires_at: datetime | None = None,
        display_name: str | None = None,
        **kwargs: Any,
    ) -> CachedContent:
        """Cache a chat-format prompt prefix for reuse by later completions on the same model.

        ``messages``, ``tools`` and ``tool_choice`` take the same shapes as ``completion``.
        ``ttl`` (seconds) and ``expires_at`` are mutually exclusive; omit both for the
        provider's default lifetime.
        """
        self._validate_expiry(ttl, expires_at, required=False)
        processed_messages = None
        if messages is not None:
            processed_messages = [
                message.model_dump(exclude_none=True, exclude={"reasoning"})
                if isinstance(message, ChatCompletionMessage)
                else message
                for message in messages
            ]
        prepared_tools = prepare_tools(tools, built_in_tools=self.BUILT_IN_TOOLS) if tools else None
        return await self._acreate_cache(
            model,
            messages=processed_messages,
            tools=prepared_tools,
            tool_choice=tool_choice,
            ttl=ttl,
            expires_at=expires_at,
            display_name=display_name,
            **kwargs,
        )

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
        message = "Provider does not support context cache creation"
        raise NotImplementedError(message)

    @handle_exceptions()
    async def alist_caches(self, *, limit: int | None = None, cursor: str | None = None, **kwargs: Any) -> CachePage:
        """Retrieve one page using an opaque cursor; never automatically fetch subsequent pages."""
        return await self._alist_caches(limit=limit, cursor=cursor, **kwargs)

    async def _alist_caches(self, *, limit: int | None = None, cursor: str | None = None, **kwargs: Any) -> CachePage:
        message = "Provider does not support context cache listing"
        raise NotImplementedError(message)

    @handle_exceptions(cache_operation=True)
    async def aretrieve_cache(self, cache_id: str, **kwargs: Any) -> CachedContent:
        """Retrieve cache metadata; the cached content itself is not returned."""
        return await self._aretrieve_cache(cache_id, **kwargs)

    async def _aretrieve_cache(self, cache_id: str, **kwargs: Any) -> CachedContent:
        message = "Provider does not support context cache retrieval"
        raise NotImplementedError(message)

    @handle_exceptions(cache_operation=True)
    async def aupdate_cache(
        self, cache_id: str, *, ttl: int | None = None, expires_at: datetime | None = None, **kwargs: Any
    ) -> CachedContent:
        """Change a cache's expiry; exactly one of ``ttl`` (seconds from now) or ``expires_at`` is required."""
        self._validate_expiry(ttl, expires_at, required=True)
        return await self._aupdate_cache(cache_id, ttl=ttl, expires_at=expires_at, **kwargs)

    async def _aupdate_cache(
        self, cache_id: str, *, ttl: int | None = None, expires_at: datetime | None = None, **kwargs: Any
    ) -> CachedContent:
        message = "Provider does not support context cache updates"
        raise NotImplementedError(message)

    @handle_exceptions(cache_operation=True)
    async def adelete_cache(self, cache_id: str, **kwargs: Any) -> CacheDeleted:
        """Delete a cache on its originating provider account."""
        return await self._adelete_cache(cache_id, **kwargs)

    async def _adelete_cache(self, cache_id: str, **kwargs: Any) -> CacheDeleted:
        message = "Provider does not support context cache deletion"
        raise NotImplementedError(message)

    def create_cache(
        self,
        model: str,
        *,
        messages: list[dict[str, Any] | ChatCompletionMessage] | None = None,
        tools: list[dict[str, Any] | Callable[..., Any]] | Any | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        ttl: int | None = None,
        expires_at: datetime | None = None,
        display_name: str | None = None,
        **kwargs: Any,
    ) -> CachedContent:
        """Run the synchronous counterpart of :meth:`acreate_cache`."""
        allow = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(
            self.acreate_cache(
                model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                ttl=ttl,
                expires_at=expires_at,
                display_name=display_name,
                **kwargs,
            ),
            allow_running_loop=allow,
        )

    def list_caches(self, *, limit: int | None = None, cursor: str | None = None, **kwargs: Any) -> CachePage:
        """Run the synchronous counterpart of :meth:`alist_caches`."""
        allow = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(self.alist_caches(limit=limit, cursor=cursor, **kwargs), allow_running_loop=allow)

    def retrieve_cache(self, cache_id: str, **kwargs: Any) -> CachedContent:
        """Run the synchronous counterpart of :meth:`aretrieve_cache`."""
        allow = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(self.aretrieve_cache(cache_id, **kwargs), allow_running_loop=allow)

    def update_cache(
        self, cache_id: str, *, ttl: int | None = None, expires_at: datetime | None = None, **kwargs: Any
    ) -> CachedContent:
        """Run the synchronous counterpart of :meth:`aupdate_cache`."""
        allow = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(
            self.aupdate_cache(cache_id, ttl=ttl, expires_at=expires_at, **kwargs), allow_running_loop=allow
        )

    def delete_cache(self, cache_id: str, **kwargs: Any) -> CacheDeleted:
        """Run the synchronous counterpart of :meth:`adelete_cache`."""
        allow = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        return run_async_in_sync(self.adelete_cache(cache_id, **kwargs), allow_running_loop=allow)
