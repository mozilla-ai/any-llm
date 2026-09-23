# ruff: noqa: PT012
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from google.genai import types

from any_llm import AnyLLM, ProviderCacheNotFoundError
from any_llm.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    UnsupportedParameterError,
)
from any_llm.providers.gemini.caches import validate_cache_id
from any_llm.providers.gemini.gemini import GeminiProvider
from any_llm.providers.vertexai.vertexai import VertexaiProvider
from any_llm.utils.aio import run_async_in_sync

CACHE = {
    "name": "cachedContents/abc",
    "model": "models/gemini-2.5-flash",
    "displayName": "docs",
    "createTime": "2026-09-23T12:00:00Z",
    "updateTime": "2026-09-23T12:00:00Z",
    "expireTime": "2026-09-23T13:00:00Z",
    "usageMetadata": {"totalTokenCount": 4096, "textCount": 12},
}
EXPIRES = datetime(2026, 9, 24, 12, tzinfo=UTC)


def provider_for(handler: Callable[[httpx.Request], httpx.Response]) -> GeminiProvider:
    return GeminiProvider(
        api_key="test-key",
        api_base="https://caches.test",
        http_options=types.HttpOptions(
            client_args={"transport": httpx.MockTransport(handler)},
            async_client_args={"transport": httpx.MockTransport(handler)},
        ),
    )


async def close_provider(provider: GeminiProvider) -> None:
    await provider.client.aio.aclose()
    provider.client.close()


def recording(requests: list[httpx.Request], body: dict[str, Any] | None = None) -> Any:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=CACHE if body is None else body)

    return handle


def get_weather(city: str) -> str:
    """Get the weather for a city.

    Args:
        city: The city name.
    """
    return city


@pytest.mark.asyncio
async def test_create_converts_messages_tools_and_ttl_like_a_completion() -> None:
    requests: list[httpx.Request] = []
    provider = provider_for(recording(requests))
    try:
        cache = await provider.acreate_cache(
            "gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "developer", "content": "Cite sources."},
                {"role": "user", "content": "Here is the manual."},
            ],
            tools=[get_weather],
            tool_choice="required",
            ttl=3600,
            display_name="docs",
            timeout=5,
        )
    finally:
        await close_provider(provider)
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.url.path.endswith("/cachedContents")
    assert request.extensions["timeout"]["read"] == 5
    body = json.loads(request.content)
    assert body["model"] == "models/gemini-2.5-flash"
    assert body["ttl"] == "3600s"
    assert body["displayName"] == "docs"
    assert body["systemInstruction"]["parts"][0]["text"] == "Be terse.\nCite sources."
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Here is the manual."}]}]
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "get_weather"
    assert body["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"
    assert cache.id == "cachedContents/abc"
    assert cache.model == "models/gemini-2.5-flash"
    assert cache.display_name == "docs"
    assert cache.created_at == datetime(2026, 9, 23, 12, tzinfo=UTC)
    assert cache.updated_at == datetime(2026, 9, 23, 12, tzinfo=UTC)
    assert cache.expires_at == datetime(2026, 9, 23, 13, tzinfo=UTC)
    assert cache.usage is not None
    assert cache.usage.total_tokens == 4096
    assert cache.usage.model_extra == {"text_count": 12}


@pytest.mark.asyncio
async def test_create_with_expires_at_and_no_messages_sends_only_the_expiry() -> None:
    requests: list[httpx.Request] = []
    provider = provider_for(recording(requests))
    try:
        await provider.acreate_cache("models/gemini-2.5-flash", expires_at=EXPIRES)
    finally:
        await close_provider(provider)
    body = json.loads(requests[0].content)
    assert body == {"model": "models/gemini-2.5-flash", "expireTime": "2026-09-24T12:00:00+00:00"}


@pytest.mark.asyncio
async def test_create_with_only_a_system_message_sends_no_contents() -> None:
    requests: list[httpx.Request] = []
    provider = provider_for(recording(requests))
    try:
        await provider.acreate_cache("gemini-2.5-flash", messages=[{"role": "system", "content": "Be terse."}])
    finally:
        await close_provider(provider)
    body = json.loads(requests[0].content)
    assert "contents" not in body
    assert body["systemInstruction"]["parts"][0]["text"] == "Be terse."


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"kms_key_name": "key"}, {"page_size": 1}, {"system_instruction": "x"}])
async def test_create_rejects_native_options_before_network(kwargs: dict[str, Any]) -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid options reached the provider"))
    try:
        with pytest.raises(UnsupportedParameterError):
            await provider.acreate_cache("gemini-2.5-flash", **kwargs)
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("next_token", [None, "page_next"])
async def test_list_returns_one_page_without_auto_pagination(next_token: str | None) -> None:
    requests: list[httpx.Request] = []
    body: dict[str, Any] = {"cachedContents": [CACHE]}
    if next_token is not None:
        body["nextPageToken"] = next_token
    provider = provider_for(recording(requests, body))
    try:
        page = await provider.alist_caches(limit=1, cursor="page_before")
    finally:
        await close_provider(provider)
    assert [cache.id for cache in page.data] == ["cachedContents/abc"]
    assert page.next_cursor == next_token
    assert len(requests) == 1
    assert requests[0].url.params["pageSize"] == "1"
    assert requests[0].url.params["pageToken"] == "page_before"


@pytest.mark.asyncio
async def test_list_without_options_sends_no_paging_parameters() -> None:
    requests: list[httpx.Request] = []
    provider = provider_for(recording(requests, {}))
    try:
        page = await provider.alist_caches()
    finally:
        await close_provider(provider)
    assert page.data == []
    assert page.next_cursor is None
    assert "pageSize" not in requests[0].url.params


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [{"page_size": 1}, {"page_token": "x"}, {"limit": 0}, {"limit": True}, {"cursor": ""}, {"timeout": 0}],
)
async def test_invalid_list_options_fail_before_network(kwargs: dict[str, Any]) -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid options reached the provider"))
    try:
        with pytest.raises((UnsupportedParameterError, InvalidRequestError)):
            await provider.alist_caches(**kwargs)
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_retrieve_canonicalizes_a_bare_id_and_forwards_headers() -> None:
    requests: list[httpx.Request] = []
    provider = provider_for(recording(requests))
    try:
        cache = await provider.aretrieve_cache("abc", extra_headers={"x-test": "1"})
    finally:
        await close_provider(provider)
    assert cache.id == "cachedContents/abc"
    assert requests[0].method == "GET"
    assert requests[0].url.path.endswith("/cachedContents/abc")
    assert requests[0].headers["x-test"] == "1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "field", "value"),
    [({"ttl": 60}, "ttl", "60s"), ({"expires_at": EXPIRES}, "expireTime", "2026-09-24T12:00:00+00:00")],
)
async def test_update_sends_exactly_the_requested_expiry(options: dict[str, Any], field: str, value: str) -> None:
    requests: list[httpx.Request] = []
    provider = provider_for(recording(requests))
    try:
        cache = await provider.aupdate_cache("cachedContents/abc", **options)
    finally:
        await close_provider(provider)
    assert cache.id == "cachedContents/abc"
    assert requests[0].method == "PATCH"
    assert requests[0].url.path.endswith("/cachedContents/abc")
    assert json.loads(requests[0].content) == {field: value}


@pytest.mark.asyncio
async def test_delete_acknowledges_with_canonical_id_without_inventing_deleted() -> None:
    requests: list[httpx.Request] = []
    provider = provider_for(recording(requests, {}))
    try:
        result = await provider.adelete_cache("abc")
    finally:
        await close_provider(provider)
    assert result.id == "cachedContents/abc"
    assert result.deleted is None
    assert requests[0].method == "DELETE"
    assert requests[0].url.path.endswith("/cachedContents/abc")


@pytest.mark.parametrize(
    ("cache_id", "expected"),
    [
        ("abc", "cachedContents/abc"),
        ("cachedContents/abc", "cachedContents/abc"),
        ("projects/p/locations/us-central1/cachedContents/abc", "projects/p/locations/us-central1/cachedContents/abc"),
    ],
)
def test_valid_cache_ids_are_canonicalized(cache_id: str, expected: str) -> None:
    assert validate_cache_id(cache_id, "gemini") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cache_id",
    [
        "",
        ".",
        "..",
        "cachedContents/",
        "cachedContents/..",
        "files/abc",
        "cachedContents/abc/extra",
        "../cachedContents/abc",
        "a b",
        "abc?x=1",
        "abc#frag",
        "abc%2F",
        "abc:delete",
        "a\\b",
        "projects//locations/l/cachedContents/abc",
        "projects/p/locations/../cachedContents/abc",
        "projects/p/regions/l/cachedContents/abc",
        123,
    ],
)
async def test_invalid_cache_ids_are_rejected_before_network(cache_id: Any) -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid cache ID reached the provider"))
    try:
        for call in (
            provider.aretrieve_cache(cache_id),
            provider.aupdate_cache(cache_id, ttl=60),
            provider.adelete_cache(cache_id),
        ):
            with pytest.raises(InvalidRequestError, match="cache ID"):
                await call
    finally:
        await close_provider(provider)


MISSING_CACHE_403 = {
    "error": {"code": 403, "message": "CachedContent not found (or permission denied)", "status": "PERMISSION_DENIED"}
}


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["retrieve", "update", "delete"])
@pytest.mark.parametrize(
    ("status", "body"),
    [
        (404, {"error": {"code": 404, "message": "Cache not found", "status": "NOT_FOUND"}}),
        (403, MISSING_CACHE_403),
    ],
)
async def test_missing_cache_has_cache_specific_error(
    operation: str, status: int, body: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(lambda _: httpx.Response(status, json=body))
    try:
        with pytest.raises(ProviderCacheNotFoundError, match="not found") as error:
            if operation == "retrieve":
                await provider.aretrieve_cache("missing")
            elif operation == "update":
                await provider.aupdate_cache("missing", ttl=60)
            else:
                await provider.adelete_cache("missing")
        assert error.value.status_code == status
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["retrieve", "update", "delete"])
async def test_permission_403_without_not_found_stays_authentication(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    body = {"error": {"code": 403, "message": "Permission denied", "status": "PERMISSION_DENIED"}}
    provider = provider_for(lambda _: httpx.Response(403, json=body))
    try:
        with pytest.raises(AuthenticationError):
            if operation == "retrieve":
                await provider.aretrieve_cache("abc")
            elif operation == "update":
                await provider.aupdate_cache("abc", ttl=60)
            else:
                await provider.adelete_cache("abc")
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_list_404_is_not_a_missing_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    body = {"error": {"code": 404, "message": "Not found", "status": "NOT_FOUND"}}
    provider = provider_for(lambda _: httpx.Response(404, json=body))
    try:
        with pytest.raises(Exception) as error:  # noqa: PT011
            await provider.alist_caches()
        assert not isinstance(error.value, ProviderCacheNotFoundError)
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_missing_name_is_a_provider_error() -> None:
    provider = provider_for(lambda _: httpx.Response(200, json={"model": "models/gemini-2.5-flash"}))
    try:
        with pytest.raises(ProviderError, match="missing a resource name"):
            await provider.aretrieve_cache("abc")
    finally:
        await close_provider(provider)


def test_sync_operations_use_the_same_contract() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        if request.method == "GET" and request.url.path.endswith("/cachedContents"):
            return httpx.Response(200, json={"cachedContents": [CACHE]})
        return httpx.Response(200, json=CACHE)

    provider = provider_for(handle)
    try:
        assert provider.create_cache("gemini-2.5-flash", ttl=60).id == "cachedContents/abc"
        assert provider.list_caches(limit=1).next_cursor is None
        assert provider.retrieve_cache("abc").usage is not None
        assert provider.update_cache("abc", ttl=60).display_name == "docs"
        assert provider.delete_cache("abc").id == "cachedContents/abc"
    finally:
        run_async_in_sync(provider.client.aio.aclose())


def test_cache_capabilities_cover_both_google_providers() -> None:
    operations = {"create", "list", "retrieve", "update", "delete"}
    for provider_class in (GeminiProvider, VertexaiProvider):
        metadata = provider_class.get_provider_metadata()
        assert metadata.caches is True
        assert set(metadata.cache_operations) == operations
    assert AnyLLM.get_provider_class("deepseek").get_provider_metadata().caches is False


@pytest.mark.asyncio
async def test_vertex_uses_the_same_caches_client_and_full_resource_names() -> None:
    vertex_name = "projects/p/locations/us-central1/cachedContents/abc"
    returned = types.CachedContent(
        name=vertex_name, model="projects/p/locations/us-central1/publishers/google/models/x"
    )
    with patch("any_llm.providers.vertexai.vertexai.genai.Client") as client_class:
        client = client_class.return_value
        client._api_client.project = "p"
        client._api_client.location = "us-central1"
        caches = Mock()
        caches.create = AsyncMock(return_value=returned)
        caches.get = AsyncMock(return_value=returned)
        caches.delete = AsyncMock()
        client.aio.caches = caches
        provider = VertexaiProvider()
        created = await provider.acreate_cache("gemini-2.5-flash", messages=[{"role": "user", "content": "hi"}], ttl=60)
        retrieved = await provider.aretrieve_cache(vertex_name)
        deleted = await provider.adelete_cache(vertex_name)
    assert created.id == vertex_name
    assert retrieved.id == vertex_name
    assert deleted.id == vertex_name
    create_kwargs = caches.create.await_args.kwargs
    assert create_kwargs["model"] == "gemini-2.5-flash"
    assert create_kwargs["config"].ttl == "60s"
    caches.get.assert_awaited_once_with(name=vertex_name, config=None)
    caches.delete.assert_awaited_once_with(name=vertex_name, config=None)


@pytest.mark.asyncio
async def test_vertex_errors_name_the_vertex_provider() -> None:
    with patch("any_llm.providers.vertexai.vertexai.genai.Client") as client_class:
        client = client_class.return_value
        client._api_client.project = "p"
        client._api_client.location = "us-central1"
        provider = VertexaiProvider()
        with pytest.raises(UnsupportedParameterError) as error:
            await provider.alist_caches(page_size=1)
    assert error.value.provider_name == "vertexai"
