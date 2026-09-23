from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from any_llm import AnyLLM, CachedContent, CacheDeleted, CachePage, ProviderCacheNotFoundError
from any_llm._caches import CachesMixin
from any_llm.exceptions import InvalidRequestError
from any_llm.types.completion import ChatCompletionMessage, Reasoning
from any_llm.utils.exception_handler import _handle_exception

EXPIRES = datetime(2026, 9, 24, 12, tzinfo=UTC)


def make_provider() -> CachesMixin:
    provider = CachesMixin()
    provider.PROVIDER_NAME = "test"
    return provider


def get_weather(city: str) -> str:
    """Get the weather for a city.

    Args:
        city: The city name.
    """
    return city


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_public_create_normalizes_messages_and_tools(synchronous: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider()
    create = AsyncMock(return_value=CachedContent(id="cachedContents/abc"))
    monkeypatch.setattr(provider, "_acreate_cache", create)
    messages: list[Any] = [
        {"role": "system", "content": "Be terse."},
        ChatCompletionMessage(role="assistant", content="Hello", reasoning=Reasoning(content="hidden")),
    ]
    options: dict[str, Any] = {
        "messages": messages,
        "tools": [get_weather],
        "tool_choice": "auto",
        "ttl": 600,
        "display_name": "docs",
        "custom_option": "value",
    }
    if synchronous:
        result = provider.create_cache("model-x", allow_running_loop=True, **options)
    else:
        result = await provider.acreate_cache("model-x", **options)
    assert result.id == "cachedContents/abc"
    args = create.await_args
    assert args is not None
    assert args.args == ("model-x",)
    kwargs = args.kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "Be terse."},
        {"role": "assistant", "content": "Hello"},
    ]
    assert kwargs["tools"][0]["function"]["name"] == "get_weather"
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["ttl"] == 600
    assert kwargs["expires_at"] is None
    assert kwargs["display_name"] == "docs"
    assert kwargs["custom_option"] == "value"


@pytest.mark.asyncio
async def test_public_create_passes_none_when_messages_and_tools_are_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider()
    create = AsyncMock(return_value=CachedContent(id="cachedContents/abc"))
    monkeypatch.setattr(provider, "_acreate_cache", create)
    await provider.acreate_cache("model-x", expires_at=EXPIRES)
    create.assert_awaited_once_with(
        "model-x", messages=None, tools=None, tool_choice=None, ttl=None, expires_at=EXPIRES, display_name=None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"ttl": 60, "expires_at": EXPIRES}, "mutually exclusive"),
        ({"ttl": 0}, "ttl must be"),
        ({"ttl": -5}, "ttl must be"),
        ({"ttl": True}, "ttl must be"),
        ({"ttl": 1.5}, "ttl must be"),
        ({"expires_at": datetime(2026, 9, 24, 12)}, "timezone-aware"),  # noqa: DTZ001
        ({"expires_at": "2026-09-24T12:00:00Z"}, "timezone-aware"),
    ],
)
async def test_invalid_create_expiry_fails_before_provider(
    options: dict[str, Any], match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = make_provider()
    create = AsyncMock()
    monkeypatch.setattr(provider, "_acreate_cache", create)
    with pytest.raises(InvalidRequestError, match=match):
        await provider.acreate_cache("model-x", **options)
    create.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({}, "Exactly one"),
        ({"ttl": 60, "expires_at": EXPIRES}, "mutually exclusive"),
        ({"ttl": 0}, "ttl must be"),
    ],
)
async def test_invalid_update_expiry_fails_before_provider(
    options: dict[str, Any], match: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = make_provider()
    update = AsyncMock()
    monkeypatch.setattr(provider, "_aupdate_cache", update)
    with pytest.raises(InvalidRequestError, match=match):
        await provider.aupdate_cache("abc", **options)
    update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_public_lifecycle_forwards_to_provider_hooks(synchronous: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider()
    cache = CachedContent(id="cachedContents/abc")
    listing = AsyncMock(side_effect=[CachePage(data=[cache], next_cursor="opaque"), CachePage(data=[])])
    retrieve = AsyncMock(return_value=cache)
    update = AsyncMock(return_value=cache)
    delete = AsyncMock(return_value=CacheDeleted(id="cachedContents/abc"))
    monkeypatch.setattr(provider, "_alist_caches", listing)
    monkeypatch.setattr(provider, "_aretrieve_cache", retrieve)
    monkeypatch.setattr(provider, "_aupdate_cache", update)
    monkeypatch.setattr(provider, "_adelete_cache", delete)
    if synchronous:
        first = provider.list_caches(limit=1, allow_running_loop=True)
        second = provider.list_caches(limit=1, cursor=first.next_cursor, allow_running_loop=True)
        provider.retrieve_cache("abc", allow_running_loop=True)
        provider.update_cache("abc", ttl=60, allow_running_loop=True)
        deleted = provider.delete_cache("abc", allow_running_loop=True)
    else:
        first = await provider.alist_caches(limit=1)
        second = await provider.alist_caches(limit=1, cursor=first.next_cursor)
        await provider.aretrieve_cache("abc")
        await provider.aupdate_cache("abc", ttl=60)
        deleted = await provider.adelete_cache("abc")
    assert second.next_cursor is None
    assert listing.await_args_list[1].kwargs == {"limit": 1, "cursor": "opaque"}
    retrieve.assert_awaited_once_with("abc")
    update.assert_awaited_once_with("abc", ttl=60, expires_at=None)
    delete.assert_awaited_once_with("abc")
    assert deleted.deleted is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "call"),
    [
        ("creation", lambda p: p.acreate_cache("model-x")),
        ("listing", lambda p: p.alist_caches()),
        ("retrieval", lambda p: p.aretrieve_cache("abc")),
        ("updates", lambda p: p.aupdate_cache("abc", ttl=60)),
        ("deletion", lambda p: p.adelete_cache("abc")),
    ],
)
async def test_unsupported_provider_raises_not_implemented(operation: str, call: Any) -> None:
    provider = AnyLLM.create("deepseek", api_key="test-key")
    assert provider.get_provider_metadata().caches is False
    assert provider.get_provider_metadata().cache_operations == ()
    with pytest.raises(NotImplementedError, match=f"context cache {operation}"):
        await call(provider)


def test_cache_404_becomes_cache_not_found_only_for_cache_operations() -> None:
    class NotFoundError(Exception):
        status_code = 404

    with pytest.raises(ProviderCacheNotFoundError) as error:
        _handle_exception(NotFoundError("gone"), "test", cache_operation=True, unified_exceptions=True)
    assert error.value.status_code == 404
    with pytest.raises(Exception) as other:  # noqa: PT011
        _handle_exception(NotFoundError("gone"), "test", unified_exceptions=True)
    assert not isinstance(other.value, ProviderCacheNotFoundError)


def test_cache_types_are_provider_neutral_and_preserve_extras() -> None:
    cache = CachedContent.model_validate({"id": "opaque", "native_field": "kept", "usage": {"total_tokens": 5}})
    assert cache.model_extra == {"native_field": "kept"}
    assert cache.usage is not None
    assert cache.usage.total_tokens == 5
    assert set(CachePage.model_fields) == {"data", "next_cursor"}
    assert set(CacheDeleted.model_fields) == {"id", "deleted"}
