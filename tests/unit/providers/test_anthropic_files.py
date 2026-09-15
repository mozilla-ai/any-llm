# ruff: noqa: PT012
import asyncio
import contextvars
import warnings
from collections.abc import AsyncIterator, Callable
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pytest
from anthropic import APIStatusError
from typing_extensions import override

from any_llm import AnyLLM, AsyncFileDownload, FileDownload
from any_llm.exceptions import (
    AnyLLMError,
    InvalidRequestError,
    ProviderError,
    ProviderFileNotFoundError,
    RateLimitError,
    UnsupportedParameterError,
)
from any_llm.providers.anthropic.anthropic import AnthropicProvider
from any_llm.utils.aio import run_async_in_sync

META = {
    "id": "file_123",
    "type": "file",
    "filename": "input.csv",
    "mime_type": "text/csv",
    "size_bytes": 4,
    "created_at": "2026-09-14T12:00:00Z",
    "downloadable": False,
    "expires_at": None,
}


def provider_for(handler: Callable[[httpx.Request], httpx.Response]) -> AnthropicProvider:
    return AnthropicProvider(
        api_key="test-key",
        api_base="https://files.test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
async def test_upload_preserves_metadata_and_sends_multipart(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    path.write_bytes(b"a,b\n")
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={**META, "future_field": "preserved"})

    provider = provider_for(handle)
    try:
        result = await provider.aupload_file(path, mime_type="text/csv", expires_in=3600)
        assert result.id == "file_123"
        assert result.size_bytes == 4
        assert result.downloadable is False
        assert result.model_extra == {"type": "file", "future_field": "preserved"}
        assert len(requests) == 1
        assert requests[0].url.path == "/v1/files"
        assert requests[0].headers["x-api-key"] == "test-key"
        assert requests[0].headers["content-type"].startswith("multipart/form-data;")
        assert b"a,b\n" in requests[0].content
        assert b'filename="input.csv"' in requests[0].content
        assert b"3600" in requests[0].content
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_retrieve_leaves_missing_metadata_unknown() -> None:
    provider = provider_for(lambda _: httpx.Response(200, json={"id": "file_123", "type": "file"}))
    try:
        result = await provider.aretrieve_file("file_123")
        assert result.downloadable is None
        assert result.expires_at is None
        assert result.size_bytes is None
    finally:
        await provider.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("next_cursor", [None, "page_next"])
async def test_list_returns_one_page_without_auto_pagination(next_cursor: str | None) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [META], "next_page": next_cursor, "future": "value"})

    provider = provider_for(handle)
    try:
        result = await provider.alist_files(limit=1, cursor="page_before")
        assert result.data[0].size_bytes == 4
        assert result.next_cursor == next_cursor
        assert result.model_extra == {"future": "value"}
        assert len(requests) == 1
        assert requests[0].url.params == httpx.QueryParams(limit=1, page="page_before")
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_delete_returns_provider_id() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": "file_123", "type": "file_deleted"})

    provider = provider_for(handle)
    try:
        result = await provider.adelete_file("file_123")
        assert result.id == "file_123"
        assert requests[0].method == "DELETE"
        assert requests[0].url.path == "/v1/files/file_123"
    finally:
        await provider.client.close()


def test_file_capabilities_are_explicit() -> None:
    anthropic = AnthropicProvider.get_provider_metadata()
    assert anthropic.files is True
    assert set(anthropic.file_operations) == {"upload", "list", "retrieve", "download", "delete"}
    assert AnyLLM.get_provider_class("openai").get_provider_metadata().files is False


@pytest.mark.asyncio
async def test_unsupported_provider_does_not_call_network() -> None:
    provider = AnyLLM.create("openai", api_key="test")
    with pytest.raises(NotImplementedError, match="file"):
        await provider.aupload_file(b"data")


class CountingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.reads = 0
        self.closed = False

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(100):
            self.reads += 1
            yield b"data"

    @override
    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_download_is_lazy_and_closes_after_early_exit() -> None:
    stream = CountingStream()
    provider = provider_for(lambda _: httpx.Response(200, stream=stream))
    try:
        async with provider.adownload_file("file_123", chunk_size=4) as chunks:
            assert stream.reads == 0
            assert await anext(chunks) == b"data"
            assert stream.reads == 1
        assert stream.closed
    finally:
        await provider.client.close()


def test_sync_calls_share_the_provider_and_download_without_prefetch() -> None:
    stream = CountingStream()
    provider = provider_for(
        lambda request: (
            httpx.Response(200, stream=stream)
            if request.url.path.endswith("/content")
            else httpx.Response(200, json=META)
        )
    )
    assert provider.upload_file(b"a,b\n").id == "file_123"
    assert provider.retrieve_file("file_123").size_bytes == 4
    with provider.download_file("file_123", chunk_size=4) as chunks:
        assert next(chunks) == b"data"
        assert stream.reads == 1
    assert stream.closed

    run_async_in_sync(provider.client.close())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs", [{"page": "x"}, {"limit": 0}, {"limit": True}, {"limit": "1"}, {"after_id": "x"}, {"purpose": "batch"}]
)
async def test_invalid_list_options_fail_before_network(kwargs: dict[str, Any]) -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid options reached the provider")

    provider = provider_for(handle)
    try:
        with pytest.raises(AnyLLMError, match=r"(ids|limit|not supported)"):
            await provider.alist_files(**kwargs)
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_missing_file_has_file_specific_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(
        lambda _: httpx.Response(
            404, json={"type": "error", "error": {"type": "not_found_error", "message": "File not found"}}
        )
    )
    try:
        with pytest.raises(ProviderFileNotFoundError, match="File not found") as error:
            await provider.aretrieve_file("file_missing")
        assert type(error.value).__name__ == "ProviderFileNotFoundError"
        assert error.value.status_code == 404
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_upload_does_not_retry_and_retains_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            429, headers={"retry-after": "3"}, json={"error": {"type": "rate_limit_error", "message": "Rate limited"}}
        )

    provider = provider_for(handle)
    try:
        with pytest.raises(RateLimitError) as error:
            await provider.aupload_file(b"data")
        assert len(requests) == 1
        assert error.value.retry_after == "3"
        assert error.value.status_code == 429
        assert provider.client.max_retries == 2
    finally:
        await provider.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"purpose": "batch"},
        {"expires_in": 0},
        {"expires_in": True},
        {"expires_in": "3600"},
        {"expires_in_seconds": 3600},
    ],
)
async def test_invalid_upload_options_are_not_silently_forwarded(kwargs: dict[str, Any]) -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid upload reached network"))
    try:
        with pytest.raises(AnyLLMError, match=r"(not supported|expires_in)"):
            await provider.aupload_file(b"data", **kwargs)
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_beta_headers_merge_with_explicit_and_default_betas() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [], "next_page": None})

    provider = provider_for(handle)
    provider.client = provider.client.with_options(default_headers={"anthropic-beta": "default-beta"})
    headers = {"Anthropic-Beta": "other-beta, shared-beta", "x-custom": "value"}
    try:
        await provider.alist_files(betas=["shared-beta", "new-beta"], extra_headers=headers)
        assert requests[0].headers["anthropic-beta"] == "default-beta,other-beta,shared-beta,new-beta"
        assert requests[0].headers["x-custom"] == "value"
        assert headers["Anthropic-Beta"] == "other-beta, shared-beta"
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_argument_errors_survive_unified_exceptions_without_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(lambda _: pytest.fail("Invalid options reached the provider"))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(UnsupportedParameterError, match="purpose"):
                await provider.aupload_file(b"data", purpose="batch")
            with pytest.raises(UnsupportedParameterError, match="purpose"):
                await provider.aretrieve_file("file_123", purpose="batch")
            with pytest.raises(UnsupportedParameterError, match="purpose"):
                await provider.adelete_file("file_123", purpose="batch")
            with pytest.raises(InvalidRequestError, match="limit"):
                await provider.alist_files(limit=0)
            with pytest.raises(InvalidRequestError, match="nonempty"):
                await provider.aretrieve_file("")
            with pytest.raises(InvalidRequestError, match="chunk_size"):
                async with provider.adownload_file("file_123", chunk_size=0):
                    pass
        assert [str(entry.message) for entry in caught] == []
    finally:
        await provider.client.close()


def test_sync_download_runs_in_one_context() -> None:
    """A transport that brackets its stream with a contextvar must be able to reset its token.

    The sync bridge advances the response in a single task for this reason; a bridge that used a
    fresh task per chunk fails here with "Token was created in a different Context".
    """
    active = contextvars.ContextVar("download_active", default="unset")

    class TracedStream(httpx.AsyncByteStream):
        @override
        async def __aiter__(self) -> AsyncIterator[bytes]:
            token = active.set("streaming")
            try:
                for _ in range(3):
                    yield b"data"
            finally:
                active.reset(token)

    provider = provider_for(lambda _: httpx.Response(200, stream=TracedStream()))
    try:
        with provider.download_file("file_123", chunk_size=4) as chunks:
            assert list(chunks) == [b"data"] * 3
    finally:
        run_async_in_sync(provider.client.close())

    assert active.get() == "unset"


def test_public_file_types_are_exported() -> None:
    import any_llm

    assert any_llm.FileMetadata(id="file_123").id == "file_123"
    assert any_llm.FilePage(data=[]).data == []
    assert any_llm.FileDeleted(id="file_123").id == "file_123"
    assert issubclass(any_llm.ProviderFileNotFoundError, any_llm.AnyLLMError)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["list", "retrieve", "delete", "download"])
async def test_unsupported_file_operations(operation: str) -> None:
    provider = AnyLLM.create("openai", api_key="test")
    with pytest.raises(NotImplementedError, match="file"):
        if operation == "list":
            await provider.alist_files()
        elif operation == "retrieve":
            await provider.aretrieve_file("file_123")
        elif operation == "delete":
            await provider.adelete_file("file_123")
        else:
            async with provider.adownload_file("file_123") as chunks:
                await anext(chunks)


@pytest.mark.asyncio
async def test_download_full_consumption_and_consumer_error_close_stream() -> None:
    for fail in (False, True):
        stream = CountingStream()

        def handle(_: httpx.Request, source: CountingStream = stream) -> httpx.Response:
            return httpx.Response(200, stream=source)

        provider = provider_for(handle)
        try:
            if fail:
                with pytest.raises(RuntimeError, match="consumer failed"):
                    async with provider.adownload_file("file_123", chunk_size=4) as chunks:
                        await anext(chunks)
                        message = "consumer failed"
                        raise RuntimeError(message)
            else:
                async with provider.adownload_file("file_123", chunk_size=4) as chunks:
                    assert len([chunk async for chunk in chunks]) == 100
            assert stream.closed
        finally:
            await provider.client.close()


@pytest.mark.asyncio
async def test_download_cancellation_closes_upstream() -> None:

    stream = CountingStream()
    provider = provider_for(lambda _: httpx.Response(200, stream=stream))
    try:
        with pytest.raises(asyncio.CancelledError):
            async with provider.adownload_file("file_123", chunk_size=4) as chunks:
                await anext(chunks)
                raise asyncio.CancelledError
        assert stream.closed
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_download_network_error_during_iteration_is_unified(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

    class BrokenStream(CountingStream):
        @override
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"data"
            message = "connection failed"
            raise httpx.ReadError(message)

    stream = BrokenStream()
    provider = provider_for(lambda _: httpx.Response(200, stream=stream))
    try:
        with pytest.raises(ProviderError, match="connection failed"):
            async with provider.adownload_file("file_123", chunk_size=4) as chunks:
                assert await anext(chunks) == b"data"
                await anext(chunks)
        assert stream.closed
    finally:
        await provider.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("file_id", ["", ".", "..", "../models", "file/../../x", "file\\x", " file_abc", "file_abc\n"])
async def test_invalid_file_ids_are_rejected(file_id: str) -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid file ID reached network"))
    try:
        with pytest.raises(InvalidRequestError, match="file ID"):
            await provider.aretrieve_file(file_id)
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_timeout_is_forwarded_and_upload_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        message = "response lost"
        raise httpx.ReadTimeout(message, request=request)

    provider = provider_for(handle)
    try:
        with pytest.raises(ProviderError):
            await provider.aupload_file(b"data", timeout=0.5)
        assert len(requests) == 1
        assert requests[0].extensions["timeout"]["read"] == 0.5
    finally:
        await provider.client.close()


def test_sync_list_delete_and_complete_download() -> None:

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": "file_123", "type": "file_deleted"})
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"abc")
        return httpx.Response(200, json={"data": [META], "next_page": None})

    provider = provider_for(handle)
    try:
        assert provider.list_files(limit=1).data[0].id == "file_123"
        assert provider.delete_file("file_123").id == "file_123"
        with provider.download_file("file_123", chunk_size=1) as chunks:
            assert list(chunks) == [b"a", b"b", b"c"]
    finally:
        run_async_in_sync(provider.client.close())


@pytest.mark.asyncio
async def test_file_handle_upload_is_bounded_and_does_not_close_callers_handle() -> None:

    class BoundedFile(BytesIO):
        @override
        def read(self, size: int | None = -1) -> bytes:
            assert size is not None
            assert 0 < size <= 65536
            return super().read(size)

    content = BoundedFile(b"x" * 200000)
    provider = provider_for(lambda _: httpx.Response(200, json=META))
    try:
        await provider.aupload_file(content, filename="data.bin")
        assert not content.closed
    finally:
        content.close()
        await provider.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["retrieve", "delete", "download"])
async def test_unknown_options_rejected(operation: str) -> None:
    provider = provider_for(lambda _: pytest.fail("Unknown option reached network"))
    try:
        with pytest.raises(UnsupportedParameterError, match="unknown"):
            if operation == "retrieve":
                await provider.aretrieve_file("file_123", unknown=True)
            elif operation == "delete":
                await provider.adelete_file("file_123", unknown=True)
            else:
                async with provider.adownload_file("file_123", unknown=True) as chunks:
                    await anext(chunks)
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_invalid_download_chunk_size() -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid chunk size reached network"))
    try:
        with pytest.raises(InvalidRequestError, match="chunk_size"):
            async with provider.adownload_file("file_123", chunk_size=0):
                pass
    finally:
        await provider.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["default", "headers", "betas"])
async def test_legacy_pagination_is_rejected(source: str) -> None:
    provider = provider_for(lambda _: pytest.fail("Legacy request reached network"))
    kwargs: dict[str, Any] = {}
    if source == "default":
        provider.client = provider.client.with_options(default_headers={"anthropic-beta": "files-api-2025-04-14"})
    elif source == "headers":
        kwargs["extra_headers"] = {"Anthropic-Beta": "other, files-api-2025-04-14"}
    else:
        kwargs["betas"] = ["files-api-2025-04-14"]
    try:
        with pytest.raises(UnsupportedParameterError, match="Legacy"):
            await provider.alist_files(**kwargs)
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_server_limits_are_not_duplicated_in_client() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=META if request.method == "POST" else {"data": [], "next_page": None})

    provider = provider_for(handle)
    try:
        await provider.aupload_file(b"data", expires_in=7776001)
        await provider.alist_files(limit=1001)
        await provider.alist_files(ids=["file_123"] * 101)
        assert b"7776001" in requests[0].content
        assert requests[1].url.params["limit"] == "1001"
        assert len(requests[2].url.params.get_list("ids[]")) == 101
    finally:
        await provider.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["upload", "list"])
async def test_collection_404_is_not_a_missing_file(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(lambda _: httpx.Response(404, json={"error": {"message": "Wrong endpoint"}}))
    try:
        with pytest.raises(AnyLLMError) as error:
            if operation == "upload":
                await provider.aupload_file(b"data")
            else:
                await provider.alist_files()
        assert not isinstance(error.value, ProviderFileNotFoundError)
        assert error.value.status_code == 404
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_actual_task_cancellation_closes_stream() -> None:
    reading = asyncio.Event()

    class WaitingStream(CountingStream):
        @override
        async def __aiter__(self) -> AsyncIterator[bytes]:
            reading.set()
            await asyncio.Event().wait()
            yield b"unreachable"

    stream = WaitingStream()
    provider = provider_for(lambda _: httpx.Response(200, stream=stream))

    async def download() -> None:
        async with provider.adownload_file("file_123") as chunks:
            await anext(chunks)

    try:
        task = asyncio.create_task(download())
        await asyncio.wait_for(reading.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed
    finally:
        await provider.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("reads", [0, 1, 100])
async def test_async_download_exposes_headers_on_entry_without_reading_body(reads: int) -> None:
    stream = CountingStream()
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/csv",
                "Content-Length": "400",
                "Content-Disposition": 'attachment; filename="data.csv"',
            },
            stream=stream,
        )

    provider = provider_for(handle)
    try:
        context = provider.adownload_file("file_123", chunk_size=4)
        assert requests == []
        async with context as download:
            assert isinstance(download, AsyncFileDownload)
            assert len(requests) == 1
            assert download.status_code == 200
            assert download.headers["content-type"] == "text/csv"
            assert download.headers["Content-Length"] == "400"
            assert download.headers["content-disposition"] == 'attachment; filename="data.csv"'
            assert stream.reads == 0
            assert not stream.closed
            for _ in range(reads):
                assert await anext(download) == b"data"
            assert stream.reads == reads
        assert stream.closed
        with pytest.raises(StopAsyncIteration):
            await anext(download)
    finally:
        await provider.client.close()


@pytest.mark.parametrize("reads", [0, 1, 100])
def test_sync_download_exposes_headers_on_entry_without_reading_body(reads: int) -> None:
    stream = CountingStream()
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"Content-Type": "text/csv", "Content-Length": "400"}, stream=stream)

    provider = provider_for(handle)
    try:
        context = provider.download_file("file_123", chunk_size=4)
        assert requests == []
        with context as download:
            assert isinstance(download, FileDownload)
            assert len(requests) == 1
            assert download.status_code == 200
            assert download.headers["content-type"] == "text/csv"
            assert download.headers["Content-Length"] == "400"
            assert stream.reads == 0
            for _ in range(reads):
                assert next(download) == b"data"
            assert stream.reads == reads
        assert stream.closed
        with pytest.raises(StopIteration):
            next(download)
    finally:
        run_async_in_sync(provider.client.close())


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 404, 413, 429])
@pytest.mark.parametrize("unified", [False, True])
async def test_download_http_errors_raise_on_context_entry(
    monkeypatch: pytest.MonkeyPatch, status_code: int, unified: bool
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1" if unified else "0")
    responses: list[httpx.Response] = []

    def handle(_: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            status_code,
            headers={"retry-after": "3"},
            json={"type": "error", "error": {"type": "api_error", "message": "download refused"}},
        )
        responses.append(response)
        return response

    provider = provider_for(handle)
    try:
        for sync in (False, True):
            with pytest.raises((AnyLLMError, APIStatusError), match="download refused") as error:
                if sync:
                    with provider.download_file("file_123", max_retries=0, allow_running_loop=True):
                        pytest.fail("Failed download entered the consumer context")
                else:
                    async with provider.adownload_file("file_123", max_retries=0):
                        pytest.fail("Failed download entered the consumer context")
            assert isinstance(error.value, (AnyLLMError, APIStatusError))
            assert error.value.status_code == status_code
            assert isinstance(error.value, AnyLLMError) is unified
            if unified and status_code == 404:
                assert isinstance(error.value, ProviderFileNotFoundError)
            if unified and status_code == 429:
                assert isinstance(error.value, RateLimitError)
                assert error.value.retry_after == "3"
        assert len(responses) == 2
        assert all(response.is_closed for response in responses)
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_download_consumer_errors_are_not_unified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    streams: list[CountingStream] = []

    def handle(_: httpx.Request) -> httpx.Response:
        stream = CountingStream()
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    provider = provider_for(handle)
    failure = RuntimeError("consumer failed")
    try:
        for sync in (False, True):
            with pytest.raises(RuntimeError) as error:
                if sync:
                    with provider.download_file("file_123", allow_running_loop=True):
                        raise failure
                else:
                    async with provider.adownload_file("file_123"):
                        raise failure
            assert error.value is failure
        assert all(stream.closed and stream.reads == 0 for stream in streams)
    finally:
        await provider.client.close()


@pytest.mark.asyncio
async def test_empty_download_exposes_headers_and_closes() -> None:
    provider = provider_for(lambda _: httpx.Response(200, headers={"content-length": "0"}, content=b""))
    try:
        async with provider.adownload_file("file_123") as download:
            assert download.status_code == 200
            assert download.headers["content-length"] == "0"
            assert [chunk async for chunk in download] == []
        with provider.download_file("file_123", allow_running_loop=True) as sync_download:
            assert sync_download.status_code == 200
            assert list(sync_download) == []
    finally:
        await provider.client.close()


@pytest.mark.parametrize("reads", [0, 1])
def test_sync_download_opens_and_closes_in_same_task(reads: int) -> None:
    active = contextvars.ContextVar("download_open", default=False)
    token: contextvars.Token[bool] | None = None

    class TracedStream(CountingStream):
        @override
        async def aclose(self) -> None:
            assert token is not None
            assert active.get()
            active.reset(token)
            await super().aclose()

    stream = TracedStream()

    def handle(_: httpx.Request) -> httpx.Response:
        nonlocal token
        token = active.set(True)
        return httpx.Response(200, stream=stream)

    provider = provider_for(handle)
    try:
        with provider.download_file("file_123", chunk_size=4) as download:
            assert download.status_code == 200
            for _ in range(reads):
                assert next(download) == b"data"
        assert stream.closed
        assert not active.get()
    finally:
        run_async_in_sync(provider.client.close())
