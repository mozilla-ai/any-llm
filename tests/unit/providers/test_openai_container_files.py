# ruff: noqa: PT012
import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from openai import APIStatusError

from any_llm.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    ProviderFileNotFoundError,
    RateLimitError,
    UnsupportedParameterError,
)
from any_llm.providers.anthropic.anthropic import AnthropicProvider
from any_llm.providers.azureopenai.azureopenai import AzureopenaiProvider
from any_llm.providers.openai.openai import OpenaiProvider
from any_llm.utils.aio import run_async_in_sync
from tests.unit.providers.test_openai_files import CountingStream

CONTAINER_META = {
    "id": "cfile-test",
    "object": "container.file",
    "bytes": 8,
    "created_at": 1700000000,
    "container_id": "cntr-test",
    "path": "/mnt/data/output.csv",
    "source": "assistant",
}
PROVIDERS = ("openai", "azureopenai")


def provider_for(name: str, handler: Callable[[httpx.Request], httpx.Response]) -> OpenaiProvider | AzureopenaiProvider:
    cls = OpenaiProvider if name == "openai" else AzureopenaiProvider
    base = "https://files.test/v1" if name == "openai" else "https://resource.openai.azure.com"
    return cls(api_key="test-key", api_base=base, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def container_path(name: str, *, content: bool = False) -> str:
    prefix = "/v1" if name == "openai" else "/openai/v1"
    return f"{prefix}/containers/cntr-test/files/cfile-test" + ("/content" if content else "")


@pytest.mark.asyncio
@pytest.mark.parametrize("name", PROVIDERS)
async def test_container_metadata_preserves_native_fields_and_request_options(name: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == container_path(name)
        assert request.headers["x-test"] == "yes"
        assert request.extensions["timeout"]["read"] == 7
        return httpx.Response(200, json={**CONTAINER_META, "future_field": "preserved"})

    provider = provider_for(name, respond)
    async with provider.client:
        metadata = await provider.aretrieve_file(
            "cfile-test", container_id="cntr-test", timeout=7, extra_headers={"x-test": "yes"}, max_retries=0
        )
    assert metadata.id == "cfile-test"
    assert metadata.size_bytes == 8
    assert metadata.created_at == datetime.fromtimestamp(1700000000, UTC)
    assert metadata.filename is None
    assert metadata.mime_type is None
    assert metadata.expires_at is None
    assert metadata.model_extra == {
        "object": "container.file",
        "container_id": "cntr-test",
        "path": "/mnt/data/output.csv",
        "source": "assistant",
        "future_field": "preserved",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("reads", [0, 1, 2])
async def test_container_download_is_lazy_and_closes(name: str, reads: int) -> None:
    stream = CountingStream()

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == container_path(name, content=True)
        assert request.headers["x-test"] == "yes"
        assert request.extensions["timeout"]["read"] == 7
        return httpx.Response(200, headers={"content-type": "text/csv"}, stream=stream)

    provider = provider_for(name, respond)
    async with provider.client:
        async with provider.adownload_file(
            "cfile-test", container_id="cntr-test", chunk_size=4, timeout=7, extra_headers={"x-test": "yes"}
        ) as download:
            assert download.status_code == 200
            assert download.headers["content-type"] == "text/csv"
            assert stream.reads == 0
            for _ in range(reads):
                assert await anext(download) == b"data"
            assert stream.reads == reads
        assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("error", [ValueError("consumer error"), asyncio.CancelledError()])
async def test_container_download_closes_on_consumer_failure(name: str, error: BaseException) -> None:
    stream = CountingStream()
    provider = provider_for(name, lambda _: httpx.Response(200, stream=stream))
    async with provider.client:
        with pytest.raises(type(error)):
            async with provider.adownload_file("cfile-test", container_id="cntr-test", chunk_size=4) as download:
                await anext(download)
                raise error
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("name", PROVIDERS)
async def test_container_stream_failure_is_unified_and_closed(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    stream = CountingStream(fail=True)
    provider = provider_for(name, lambda _: httpx.Response(200, stream=stream))
    async with provider.client:
        with pytest.raises(ProviderError, match="connection lost"):
            async with provider.adownload_file("cfile-test", container_id="cntr-test") as download:
                await anext(download)
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("operation", ["retrieve", "download"])
@pytest.mark.parametrize(
    "container_id",
    ["", ".", "..", "../cntr", "cntr/bad", "cntr\\bad", "cntr?bad", "cntr#bad", "cntr%2fbad", "cntr bad", 123, False],
)
async def test_invalid_container_ids_never_reach_network(name: str, operation: str, container_id: Any) -> None:
    provider = provider_for(name, lambda _: pytest.fail("Unsafe container ID reached HTTP"))
    async with provider.client:
        with pytest.raises(InvalidRequestError):
            if operation == "retrieve":
                await provider.aretrieve_file("cfile-test", container_id=container_id)
            else:
                async with provider.adownload_file("cfile-test", container_id=container_id):
                    pytest.fail("Invalid container download opened")


@pytest.mark.asyncio
@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("operation", ["retrieve", "download"])
@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("status", [401, 404, 429])
async def test_container_errors_keep_existing_file_error_contract(
    name: str, operation: str, unified: bool, status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1" if unified else "0")
    provider = provider_for(
        name, lambda _: httpx.Response(status, headers={"retry-after": "0"}, json={"error": {"message": "failed"}})
    )
    expected = (
        {401: AuthenticationError, 404: ProviderFileNotFoundError, 429: RateLimitError}[status]
        if unified
        else APIStatusError
    )
    async with provider.client:
        with pytest.raises(expected):
            if operation == "retrieve":
                await provider.aretrieve_file("cfile-test", container_id="cntr-test", max_retries=0)
            else:
                async with provider.adownload_file("cfile-test", container_id="cntr-test", max_retries=0):
                    pytest.fail("Failed response opened download")


@pytest.mark.asyncio
@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("operation", ["retrieve", "download"])
async def test_container_requests_still_reject_unknown_options(name: str, operation: str) -> None:
    provider = provider_for(name, lambda _: pytest.fail("Unknown option reached HTTP"))
    async with provider.client:
        with pytest.raises(UnsupportedParameterError, match="unknown"):
            if operation == "retrieve":
                await provider.aretrieve_file("cfile-test", container_id="cntr-test", unknown=True)
            else:
                async with provider.adownload_file("cfile-test", container_id="cntr-test", unknown=True):
                    pytest.fail("Unknown option opened download")


@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("operation", ["retrieve", "download"])
def test_sync_container_reads_use_same_routes_and_stream_cleanup(name: str, operation: str) -> None:
    stream = CountingStream()

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == container_path(name, content=operation == "download")
        return (
            httpx.Response(200, stream=stream) if operation == "download" else httpx.Response(200, json=CONTAINER_META)
        )

    provider = provider_for(name, respond)
    try:
        if operation == "retrieve":
            assert provider.retrieve_file("cfile-test", container_id="cntr-test").size_bytes == 8
        else:
            with provider.download_file("cfile-test", container_id="cntr-test", chunk_size=4) as download:
                assert stream.reads == 0
                assert next(download) == b"data"
            assert stream.closed
    finally:
        run_async_in_sync(provider.client.close())


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["retrieve", "download"])
async def test_anthropic_does_not_accept_container_id(operation: str) -> None:
    provider = AnthropicProvider(
        api_key="test-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: pytest.fail("HTTP called"))),
    )
    async with provider.client:
        with pytest.raises(UnsupportedParameterError, match="container_id"):
            if operation == "retrieve":
                await provider.aretrieve_file("file_123", container_id="cntr-test")
            else:
                async with provider.adownload_file("file_123", container_id="cntr-test"):
                    pytest.fail("Anthropic container download opened")


@pytest.mark.asyncio
@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("operation", ["retrieve", "download"])
async def test_none_container_id_keeps_ordinary_file_routes(name: str, operation: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        prefix = "/v1" if name == "openai" else "/openai/v1"
        suffix = "/content" if operation == "download" else ""
        assert request.url.path == f"{prefix}/files/file-test{suffix}"
        return (
            httpx.Response(200, content=b"data")
            if operation == "download"
            else httpx.Response(200, json={"id": "file-test"})
        )

    provider = provider_for(name, respond)
    async with provider.client:
        if operation == "retrieve":
            assert (await provider.aretrieve_file("file-test", container_id=None)).id == "file-test"
        else:
            async with provider.adownload_file("file-test", container_id=None) as download:
                assert b"".join([chunk async for chunk in download]) == b"data"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("operation", ["retrieve", "download"])
async def test_container_retries_follow_request_override(name: str, operation: str) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == container_path(name, content=operation == "download")
        return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "rate limit"}})

    provider = provider_for(name, respond)
    async with provider.client:
        with pytest.raises(APIStatusError):
            if operation == "retrieve":
                await provider.aretrieve_file("cfile-test", container_id="cntr-test", max_retries=1)
            else:
                async with provider.adownload_file("cfile-test", container_id="cntr-test", max_retries=1):
                    pytest.fail("Rate-limited download opened")
    assert len(requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("operation", ["retrieve", "download"])
async def test_container_file_ids_cannot_change_request_path(name: str, operation: str) -> None:
    provider = provider_for(name, lambda _: pytest.fail("Unsafe file ID reached HTTP"))
    async with provider.client:
        with pytest.raises(InvalidRequestError):
            if operation == "retrieve":
                await provider.aretrieve_file("../file", container_id="cntr-test")
            else:
                async with provider.adownload_file("../file", container_id="cntr-test"):
                    pytest.fail("Invalid file ID opened download")


@pytest.mark.asyncio
@pytest.mark.parametrize("name", PROVIDERS)
@pytest.mark.parametrize("operation", ["upload", "list", "delete"])
async def test_other_file_operations_do_not_accept_container_id(name: str, operation: str) -> None:
    provider = provider_for(name, lambda _: pytest.fail("Unsupported operation reached HTTP"))
    async with provider.client:
        with pytest.raises(UnsupportedParameterError, match="container_id"):
            if operation == "upload":
                await provider.aupload_file(b"data", purpose="batch", container_id="cntr-test")
            elif operation == "list":
                await provider.alist_files(container_id="cntr-test")
            else:
                await provider.adelete_file("cfile-test", container_id="cntr-test")
