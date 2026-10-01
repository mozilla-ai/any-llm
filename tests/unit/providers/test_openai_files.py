# ruff: noqa: PT012
import asyncio
import warnings
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import httpx
import pytest
from openai import APIStatusError
from typing_extensions import override

from any_llm import AnyLLM
from any_llm.exceptions import (
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    ProviderFileNotFoundError,
    RateLimitError,
    UnsupportedParameterError,
)
from any_llm.providers.azureopenai.azureopenai import AzureopenaiProvider
from any_llm.providers.openai.custom import OpenAICompatibleProvider
from any_llm.providers.openai.openai import OpenaiProvider
from any_llm.utils.aio import run_async_in_sync

if TYPE_CHECKING:
    from any_llm.types.files import FileInput

META: dict[str, Any] = {
    "id": "file-test",
    "object": "file",
    "filename": "input.jsonl",
    "bytes": 3,
    "created_at": 1700000000,
    "expires_at": 1700003600,
    "purpose": "batch",
    "status": "processed",
}
CONTAINER_META: dict[str, Any] = {
    "id": "cfile-xyz",
    "object": "container.file",
    "bytes": 12,
    "container_id": "cntr_abc",
    "created_at": 1700000000,
    "path": "/mnt/data/result.csv",
    "source": "assistant",
}
OPERATIONS = ("upload", "list", "retrieve", "delete", "download")


def provider_for(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> OpenaiProvider:
    return OpenaiProvider(
        api_key="test-key",
        api_base="https://files.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


async def call_operation(provider: OpenaiProvider, operation: str, **kwargs: Any) -> Any:
    if operation == "upload":
        return await provider.aupload_file(b"{}\n", purpose="batch", **kwargs)
    if operation == "list":
        return await provider.alist_files(**kwargs)
    if operation == "retrieve":
        return await provider.aretrieve_file("file-test", **kwargs)
    if operation == "delete":
        return await provider.adelete_file("file-test", **kwargs)
    async with provider.adownload_file("file-test", **kwargs) as download:
        return b"".join([chunk async for chunk in download])


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["path", "string", "bytes", "handle"])
async def test_upload_normalizes_metadata_and_sends_multipart(source: str, tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_bytes(b"{}\n")
    handle = BytesIO(b"{}\n")
    inputs: dict[str, FileInput] = {"path": path, "string": str(path), "bytes": b"{}\n", "handle": handle}
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={**META, "future_field": "preserved"})

    provider = provider_for(respond)
    async with provider.client:
        result = await provider.aupload_file(inputs[source], purpose="batch", expires_in=3600)
    assert result.size_bytes == 3
    assert result.created_at == datetime.fromtimestamp(META["created_at"], UTC)
    assert result.expires_at == datetime.fromtimestamp(META["expires_at"], UTC)
    assert result.purpose == "batch"
    assert result.status == "processed"
    assert result.mime_type is None
    assert result.downloadable is None
    assert result.model_extra == {"object": "file", "future_field": "preserved"}
    assert not handle.closed
    handle.close()
    assert len(requests) == 1
    request = requests[0]
    assert request.url.path == "/v1/files"
    assert request.headers["authorization"] == "Bearer test-key"
    assert (
        b'filename="input.jsonl"' in request.content
        if source in {"path", "string"}
        else b'filename="upload"' in request.content
    )
    assert b"Content-Type: application/octet-stream" in request.content
    assert b'name="purpose"\r\n\r\nbatch' in request.content
    assert b'name="expires_after[anchor]"\r\n\r\ncreated_at' in request.content
    assert b'name="expires_after[seconds]"\r\n\r\n3600' in request.content


@pytest.mark.asyncio
async def test_upload_overrides_filename_and_mime_type_without_default_expiry() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert b'filename="report.txt"' in request.content
        assert b"Content-Type: text/plain" in request.content
        assert b"expires_after" not in request.content
        return httpx.Response(200, json=META)

    provider = provider_for(respond)
    async with provider.client:
        await provider.aupload_file(b"hello", purpose="user_data", filename="report.txt", mime_type="text/plain")


@pytest.mark.asyncio
async def test_upload_closes_owned_handle_on_error() -> None:
    handle = BytesIO(b"{}\n")
    provider = provider_for(lambda _: httpx.Response(500, json={"error": {"message": "server error"}}))
    async with provider.client:
        with patch.object(Path, "open", return_value=handle), pytest.raises(APIStatusError):
            await provider.aupload_file(Path("input.jsonl"), purpose="batch")
    assert handle.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_class", [OpenaiProvider, AzureopenaiProvider])
@pytest.mark.parametrize("unified", ["0", "1"])
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize(
    "local_error",
    [
        FileNotFoundError(2, "No such file or directory"),
        PermissionError(13, "Permission denied"),
        OSError("cannot open"),
    ],
)
async def test_upload_path_errors_are_invalid_requests(
    provider_class: type[OpenaiProvider] | type[AzureopenaiProvider],
    unified: str,
    synchronous: bool,
    local_error: OSError,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", unified)
    provider = provider_class(
        api_key="test-key",
        api_base="https://files.test/openai/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: pytest.fail("Unreadable upload path reached the network"))
        ),
    )
    async with provider.client:
        with patch.object(Path, "open", side_effect=local_error), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(InvalidRequestError, match=r"Cannot open upload path 'input\.jsonl'") as error:
                if synchronous:
                    provider.upload_file(Path("input.jsonl"), purpose="batch", allow_running_loop=True)
                else:
                    await provider.aupload_file(Path("input.jsonl"), purpose="batch")
    assert error.value.original_exception is local_error
    assert error.value.__cause__ is local_error
    assert error.value.provider_name == provider.PROVIDER_NAME
    assert error.value.status_code is None
    assert not caught


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_class", [OpenaiProvider, AzureopenaiProvider])
@pytest.mark.parametrize("unified", ["0", "1"])
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("purpose", [None, "", "  ", 123, True, 1.5, b"batch", [], {}])
async def test_upload_requires_explicit_purpose(
    provider_class: type[OpenaiProvider] | type[AzureopenaiProvider],
    unified: str,
    synchronous: bool,
    purpose: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", unified)

    def unexpected(_: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid purpose must fail before network access")

    provider = provider_class(
        api_key="test-key",
        api_base="https://files.test/openai/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(unexpected)),
    )
    async with provider.client:
        with pytest.raises(InvalidRequestError, match="purpose is required") as error:
            if synchronous:
                provider.upload_file(b"data", purpose=purpose, allow_running_loop=True)
            else:
                await provider.aupload_file(b"data", purpose=purpose)
    assert error.value.provider_name == provider.PROVIDER_NAME


@pytest.mark.asyncio
@pytest.mark.parametrize("expires_in", [0, -1, True, 1.5, "3600"])
async def test_upload_rejects_invalid_expiry(expires_in: Any) -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid expiry reached the network"))
    async with provider.client:
        with pytest.raises(InvalidRequestError, match="expires_in"):
            await provider.aupload_file(b"data", purpose="batch", expires_in=expires_in)


@pytest.mark.asyncio
async def test_retrieve_leaves_unavailable_metadata_unknown() -> None:
    provider = provider_for(lambda _: httpx.Response(200, json={"id": "file-test", "object": "file"}))
    async with provider.client:
        result = await provider.aretrieve_file("file-test")
    assert result.size_bytes is None
    assert result.created_at is None
    assert result.expires_at is None
    assert result.status is None


@pytest.mark.asyncio
@pytest.mark.parametrize("has_more", [False, True])
async def test_list_fetches_one_page_and_translates_cursor(has_more: bool) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [META], "has_more": has_more, "future": "preserved"})

    provider = provider_for(respond)
    async with provider.client:
        result = await provider.alist_files(limit=1, cursor="file-before", purpose="batch", order="asc")
    assert len(requests) == 1
    assert requests[0].url.params == httpx.QueryParams(limit=1, after="file-before", purpose="batch", order="asc")
    assert result.next_cursor == ("file-test" if has_more else None)
    assert result.data[0].size_bytes == 3
    assert result.model_extra == {"has_more": has_more, "future": "preserved"}


@pytest.mark.asyncio
async def test_empty_list_omits_optional_parameters() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert not request.url.params
        return httpx.Response(200, json={"data": [], "has_more": False})

    provider = provider_for(respond)
    async with provider.client:
        page = await provider.alist_files()
    assert page.data == []
    assert page.next_cursor is None


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [{"data": []}, {"data": [], "has_more": True}])
async def test_list_rejects_ambiguous_pagination(body: dict[str, Any]) -> None:
    provider = provider_for(lambda _: httpx.Response(200, json=body))
    async with provider.client:
        with pytest.raises(ProviderError, match="pagination information"):
            await provider.alist_files()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs", [{"limit": 0}, {"limit": -1}, {"limit": True}, {"limit": 1.5}, {"order": "bad"}, {"cursor": "../bad"}]
)
async def test_list_rejects_invalid_options(kwargs: dict[str, Any]) -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid option reached the network"))
    async with provider.client:
        with pytest.raises(InvalidRequestError):
            await provider.alist_files(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("option", ["unknown", "after", "expires_after"])
async def test_unknown_and_native_alias_options_are_rejected(operation: str, option: str) -> None:
    provider = provider_for(lambda _: pytest.fail("Unknown option reached the network"))
    async with provider.client:
        with pytest.raises(UnsupportedParameterError, match=option):
            await call_operation(provider, operation, **{option: "value"})


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["retrieve", "delete", "download"])
@pytest.mark.parametrize(
    "file_id", ["", ".", "..", "../files", "file\\bad", "file?bad", "file#bad", "file%2fbad", " file", "file name"]
)
async def test_file_ids_cannot_change_request_path(operation: str, file_id: str) -> None:
    provider = provider_for(lambda _: pytest.fail("Unsafe ID reached the network"))
    async with provider.client:
        with pytest.raises(InvalidRequestError):
            if operation == "download":
                async with provider.adownload_file(file_id):
                    pytest.fail("Download context must not open")
            else:
                await getattr(provider, f"a{operation}_file")(file_id)


@pytest.mark.asyncio
async def test_delete_preserves_acknowledgement() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert request.url.path == "/v1/files/file-test"
        return httpx.Response(200, json={"id": "file-test", "object": "file", "deleted": True, "future": 1})

    provider = provider_for(respond)
    async with provider.client:
        result = await provider.adelete_file("file-test")
    assert result.deleted is True
    assert "type" not in result.model_dump()
    assert result.model_extra == {"object": "file", "future": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("unified", [False, True])
async def test_missing_file_error_is_operation_specific(
    operation: str, unified: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1" if unified else "0")
    provider = provider_for(lambda _: httpx.Response(404, json={"error": {"message": "not found"}}), max_retries=0)
    expected = (
        (ProviderFileNotFoundError if operation in {"retrieve", "delete", "download"} else ModelNotFoundError)
        if unified
        else APIStatusError
    )
    async with provider.client:
        with pytest.raises(expected):
            await call_operation(provider, operation)


@pytest.mark.asyncio
@pytest.mark.parametrize("retries", [None, 0, 1])
async def test_upload_retry_override_and_rate_limit_metadata(
    retries: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "rate limit"}})

    provider = provider_for(respond, max_retries=3)
    options: dict[str, Any] = {} if retries is None else {"max_retries": retries}
    async with provider.client:
        with pytest.raises(RateLimitError) as error:
            await provider.aupload_file(b"data", purpose="batch", **options)
    assert len(requests) == (1 if retries is None else retries + 1)
    assert error.value.retry_after == "0"


class CountingStream(httpx.AsyncByteStream):
    def __init__(self, *, fail: bool = False) -> None:
        self.reads = 0
        self.closed = False
        self.fail = fail

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(3):
            self.reads += 1
            yield b"data"
        if self.fail:
            message = "connection lost"
            raise httpx.ReadError(message)

    @override
    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("reads", [0, 1, 3])
async def test_download_exposes_headers_without_prefetch_and_closes(reads: int) -> None:
    stream = CountingStream()

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/files/file-test/content"
        assert request.headers["x-test"] == "yes"
        assert request.extensions["timeout"]["read"] == 7
        return httpx.Response(200, headers={"content-type": "text/plain"}, stream=stream)

    provider = provider_for(respond)
    async with provider.client:
        async with provider.adownload_file(
            "file-test", chunk_size=4, timeout=7, extra_headers={"x-test": "yes"}
        ) as download:
            assert download.status_code == 200
            assert download.headers["Content-Type"] == "text/plain"
            assert stream.reads == 0
            for _ in range(reads):
                assert await anext(download) == b"data"
            assert stream.reads == reads
        assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("consumer error"), asyncio.CancelledError()])
async def test_consumer_error_and_cancellation_close_download(
    error: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    stream = CountingStream()
    provider = provider_for(lambda _: httpx.Response(200, stream=stream))
    async with provider.client:
        with pytest.raises(type(error)):
            async with provider.adownload_file("file-test", chunk_size=4) as download:
                await anext(download)
                raise error
        assert stream.closed


@pytest.mark.asyncio
async def test_stream_failure_is_unified_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    stream = CountingStream(fail=True)
    provider = provider_for(lambda _: httpx.Response(200, stream=stream))
    async with provider.client:
        with pytest.raises(ProviderError, match="connection lost"):
            await call_operation(provider, "download")
    assert stream.closed


@pytest.mark.asyncio
async def test_download_auth_error_raises_before_context_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(lambda _: httpx.Response(401, json={"error": {"message": "unauthorized"}}))
    async with provider.client:
        with pytest.raises(AuthenticationError):
            async with provider.adownload_file("file-test"):
                pytest.fail("HTTP error must be raised on entry")


def test_sync_operations_use_same_contract_and_download_is_lazy() -> None:
    stream = CountingStream()

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/content"):
            return httpx.Response(200, stream=stream)
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": "file-test", "object": "file", "deleted": True})
        if request.method == "GET" and request.url.path == "/v1/files":
            return httpx.Response(200, json={"data": [META], "has_more": False})
        return httpx.Response(200, json=META)

    provider = provider_for(respond)
    try:
        assert provider.upload_file(b"{}\n", purpose="batch").size_bytes == 3
        assert provider.retrieve_file("file-test").id == "file-test"
        assert provider.list_files().next_cursor is None
        assert provider.delete_file("file-test").deleted
        with provider.download_file("file-test", chunk_size=4) as download:
            assert download.status_code == 200
            assert stream.reads == 0
            assert next(download) == b"data"
            assert stream.reads == 1
        assert stream.closed
    finally:
        run_async_in_sync(provider.client.close())


def test_files_capabilities_are_explicit_on_openai_and_azure() -> None:
    for provider_class in (OpenaiProvider, AzureopenaiProvider):
        metadata = provider_class.get_provider_metadata()
        assert metadata.files
        assert set(metadata.file_operations) == set(OPERATIONS)
    assert not AnyLLM.get_provider_class("deepseek").get_provider_metadata().files
    custom = AnyLLM.create_openai_compatible(name="custom", api_base="https://custom.test/v1", api_key="test")
    assert isinstance(custom, OpenAICompatibleProvider)
    try:
        assert not custom.get_provider_metadata().files
    finally:
        run_async_in_sync(custom.client.close())


@pytest.mark.asyncio
async def test_container_retrieve_maps_path_and_does_not_use_files_endpoint() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=CONTAINER_META)

    provider = provider_for(respond)
    async with provider.client:
        result = await provider.aretrieve_file("cfile-xyz", container_id="cntr_abc")
    assert [request.url.path for request in requests] == ["/v1/containers/cntr_abc/files/cfile-xyz"]
    assert result.id == "cfile-xyz"
    assert result.filename == "result.csv"
    assert result.size_bytes == 12
    assert result.downloadable is True
    assert result.created_at == datetime.fromtimestamp(CONTAINER_META["created_at"], UTC)
    assert result.model_extra is not None
    assert result.model_extra["container_id"] == "cntr_abc"
    assert result.model_extra["path"] == "/mnt/data/result.csv"
    assert result.model_extra["source"] == "assistant"
    assert result.model_extra["object"] == "container.file"


@pytest.mark.asyncio
@pytest.mark.parametrize("reads", [0, 1, 3])
async def test_container_download_streams_content_without_prefetch(reads: int) -> None:
    stream = CountingStream()
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/containers/cntr_abc/files/cfile-xyz/content"
        assert request.headers["x-test"] == "yes"
        return httpx.Response(200, headers={"content-type": "text/csv"}, stream=stream)

    provider = provider_for(respond)
    async with provider.client:
        async with provider.adownload_file(
            "cfile-xyz", container_id="cntr_abc", chunk_size=4, extra_headers={"x-test": "yes"}
        ) as download:
            assert download.status_code == 200
            assert download.headers["content-type"] == "text/csv"
            assert stream.reads == 0
            for _ in range(reads):
                assert await anext(download) == b"data"
            assert stream.reads == reads
        assert stream.closed
    assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["retrieve", "download"])
async def test_container_404_is_a_missing_file(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(
        lambda _: httpx.Response(404, json={"error": {"message": "container file not found"}}), max_retries=0
    )
    async with provider.client:
        with pytest.raises(ProviderFileNotFoundError, match="container file not found"):
            if operation == "retrieve":
                await provider.aretrieve_file("cfile-xyz", container_id="cntr_abc")
            else:
                async with provider.adownload_file("cfile-xyz", container_id="cntr_abc"):
                    pytest.fail("Missing container file entered the consumer context")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["retrieve", "download"])
@pytest.mark.parametrize("container_id", ["", ".", "..", "../cntr", "cntr\\bad", " cntr", True, 1])
async def test_invalid_container_id_fails_before_network(operation: str, container_id: Any) -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid container ID reached the network"))
    async with provider.client:
        with pytest.raises(InvalidRequestError):
            if operation == "retrieve":
                await provider.aretrieve_file("cfile-xyz", container_id=container_id)
            else:
                async with provider.adownload_file("cfile-xyz", container_id=container_id):
                    pytest.fail("Invalid container ID opened a download")


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["upload", "list", "delete"])
async def test_container_id_is_rejected_on_non_read_operations(operation: str) -> None:
    provider = provider_for(lambda _: pytest.fail("container_id reached a non-read operation"))
    async with provider.client:
        with pytest.raises(UnsupportedParameterError, match="container_id"):
            await call_operation(provider, operation, container_id="cntr_abc")


def test_sync_container_retrieve_and_download() -> None:
    stream = CountingStream()
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/content"):
            return httpx.Response(200, headers={"content-type": "text/csv"}, stream=stream)
        return httpx.Response(200, json=CONTAINER_META)

    provider = provider_for(respond)
    try:
        metadata = provider.retrieve_file("cfile-xyz", container_id="cntr_abc")
        assert metadata.filename == "result.csv"
        assert metadata.downloadable is True
        with provider.download_file("cfile-xyz", container_id="cntr_abc", chunk_size=4) as download:
            assert download.status_code == 200
            assert stream.reads == 0
            assert next(download) == b"data"
            assert stream.reads == 1
        assert stream.closed
        assert [request.url.path for request in requests] == [
            "/v1/containers/cntr_abc/files/cfile-xyz",
            "/v1/containers/cntr_abc/files/cfile-xyz/content",
        ]
    finally:
        run_async_in_sync(provider.client.close())
