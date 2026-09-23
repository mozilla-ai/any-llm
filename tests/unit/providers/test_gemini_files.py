# ruff: noqa: PT012
import asyncio
import warnings
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from google.genai import types
from google.genai.errors import ClientError
from typing_extensions import override

from any_llm import AnyLLM, AsyncFileDownload, FileDownload
from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    ProviderFileNotFoundError,
    RateLimitError,
    UnsupportedParameterError,
)
from any_llm.providers.gemini.gemini import GeminiProvider
from any_llm.providers.vertexai.vertexai import VertexaiProvider
from any_llm.utils.aio import run_async_in_sync

if TYPE_CHECKING:
    from any_llm.types.files import FileInput

UPLOADED = {
    "name": "files/abc-123",
    "displayName": "input.csv",
    "mimeType": "text/csv",
    "sizeBytes": 4,
    "createTime": "2026-09-14T12:00:00Z",
    "expirationTime": "2026-09-16T12:00:00Z",
    "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc-123",
    "state": "ACTIVE",
    "source": "UPLOADED",
    "sha256Hash": "abcd",
}
GENERATED = {
    **UPLOADED,
    "name": "files/gen-1",
    "displayName": "output.txt",
    "mimeType": "text/plain",
    "downloadUri": "https://generativelanguage.googleapis.com/v1beta/files/gen-1:download?alt=media",
    "source": "GENERATED",
    "uri": "https://generativelanguage.googleapis.com/v1beta/files/gen-1",
}


def provider_for(handler: Callable[[httpx.Request], httpx.Response]) -> GeminiProvider:
    return GeminiProvider(
        api_key="test-key",
        api_base="https://files.test",
        http_options=types.HttpOptions(async_client_args={"transport": httpx.MockTransport(handler)}),
    )


def uploaded_file(**overrides: Any) -> types.File:
    payload: dict[str, Any] = {
        "name": "files/abc-123",
        "display_name": "input.csv",
        "mime_type": "text/csv",
        "size_bytes": 4,
        "create_time": datetime(2026, 9, 14, 12, tzinfo=UTC),
        "expiration_time": datetime(2026, 9, 16, 12, tzinfo=UTC),
        "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc-123",
        "state": types.FileState.ACTIVE,
        "source": types.FileSource.UPLOADED,
        "sha256_hash": "abcd",
    }
    payload.update(overrides)
    return types.File(**payload)


async def close_provider(provider: GeminiProvider) -> None:
    await provider.client.aio.aclose()


@pytest.fixture
def mock_client() -> Any:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        yield client_class.return_value


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["path", "string", "bytes", "handle"])
async def test_upload_maps_inputs_and_metadata(source: str, tmp_path: Path, mock_client: Any) -> None:
    path = tmp_path / "input.csv"
    path.write_bytes(b"a,b\n")
    handle = BytesIO(b"a,b\n")
    inputs: dict[str, FileInput] = {"path": path, "string": str(path), "bytes": b"a,b\n", "handle": handle}
    captured: list[tuple[Any, Any]] = []

    async def upload(*, file: Any, config: Any = None) -> types.File:
        captured.append((file, config))
        return uploaded_file()

    mock_client.aio.files.upload = upload
    provider = GeminiProvider(api_key="test-key")
    result = await provider.aupload_file(inputs[source], mime_type="text/csv")
    assert result.id == "files/abc-123"
    assert result.filename == "input.csv"
    assert result.size_bytes == 4
    assert result.mime_type == "text/csv"
    assert result.status == "ACTIVE"
    assert result.downloadable is False
    assert result.created_at == datetime(2026, 9, 14, 12, tzinfo=UTC)
    assert result.expires_at == datetime(2026, 9, 16, 12, tzinfo=UTC)
    assert result.model_extra is not None
    assert result.model_extra["uri"] == "https://generativelanguage.googleapis.com/v1beta/files/abc-123"
    assert result.model_extra["source"] == "UPLOADED"
    file_arg, config = captured[0]
    assert config.mime_type == "text/csv"
    assert config.http_options is not None
    assert config.http_options.retry_options is not None
    assert config.http_options.retry_options.attempts == 1
    if source in {"path", "string"}:
        # Paths are opened locally and handed over as a handle, which is closed afterwards.
        assert Path(file_arg.name) == path
        assert file_arg.closed
        assert config.display_name == "input.csv"
    else:
        assert config.display_name == "upload"
        assert not isinstance(file_arg, (str, Path))
    assert not handle.closed
    handle.close()


@pytest.mark.asyncio
async def test_upload_overrides_filename_as_display_name(mock_client: Any) -> None:
    async def upload(*, file: Any, config: Any = None) -> types.File:
        assert config.display_name == "report.txt"
        assert config.mime_type == "text/plain"
        return uploaded_file(display_name="report.txt", mime_type="text/plain")

    mock_client.aio.files.upload = upload
    provider = GeminiProvider(api_key="test-key")
    result = await provider.aupload_file(b"hello", filename="report.txt", mime_type="text/plain")
    assert result.filename == "report.txt"


@pytest.mark.asyncio
async def test_retrieve_leaves_missing_metadata_unknown() -> None:
    provider = provider_for(lambda _: httpx.Response(200, json={"name": "files/abc-123"}))
    try:
        result = await provider.aretrieve_file("files/abc-123")
        assert result.id == "files/abc-123"
        assert result.downloadable is None
        assert result.expires_at is None
        assert result.size_bytes is None
        assert result.status is None
        assert result.filename is None
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_generated_file_metadata_is_downloadable() -> None:
    provider = provider_for(lambda _: httpx.Response(200, json=GENERATED))
    try:
        result = await provider.aretrieve_file("files/gen-1")
        assert result.downloadable is True
        assert result.filename == "output.txt"
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_retrieve_normalizes_timestamps_and_preserves_native_extras() -> None:
    payload = {**UPLOADED, "createTime": "2026-09-14T12:00:00.123456+02:00"}
    provider = provider_for(lambda _: httpx.Response(200, json=payload))
    try:
        result = await provider.aretrieve_file("abc-123")
        assert result.id == "files/abc-123"
        assert result.created_at == datetime(2026, 9, 14, 10, 0, 0, 123456, tzinfo=UTC)
        assert result.model_extra is not None
        assert result.model_extra["uri"] == UPLOADED["uri"]
        assert result.model_extra["source"] == "UPLOADED"
        assert result.model_extra["sha256_hash"] == "abcd"
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("next_token", [None, "page_next"])
async def test_list_returns_one_page_without_auto_pagination(next_token: str | None) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        body: dict[str, Any] = {"files": [UPLOADED]}
        if next_token is not None:
            body["nextPageToken"] = next_token
        return httpx.Response(200, json=body)

    provider = provider_for(handle)
    try:
        result = await provider.alist_files(limit=1, cursor="page_before")
        assert result.data[0].id == "files/abc-123"
        assert result.data[0].size_bytes == 4
        assert result.next_cursor == next_token
        assert len(requests) == 1
        assert requests[0].url.params["pageSize"] == "1"
        assert requests[0].url.params["pageToken"] == "page_before"
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_delete_acknowledges_with_requested_id_without_inventing_deleted() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    provider = provider_for(handle)
    try:
        result = await provider.adelete_file("abc-123")
        assert result.id == "files/abc-123"
        assert result.deleted is None
        assert requests[0].method == "DELETE"
        assert requests[0].url.path.endswith("/files/abc-123")
    finally:
        await close_provider(provider)


def test_file_capabilities_are_gemini_only() -> None:
    gemini = GeminiProvider.get_provider_metadata()
    assert gemini.files is True
    assert set(gemini.file_operations) == {"upload", "list", "retrieve", "delete"}
    assert VertexaiProvider.get_provider_metadata().files is False
    assert AnyLLM.get_provider_class("deepseek").get_provider_metadata().files is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [{"purpose": "batch"}, {"expires_in": 3600}, {"expires_in_seconds": 3600}, {"page_size": 1}],
)
async def test_invalid_upload_options_fail_before_sdk(kwargs: dict[str, Any], mock_client: Any) -> None:
    mock_client.aio.files.upload = AsyncMock(side_effect=AssertionError("Invalid upload reached the SDK"))
    provider = GeminiProvider(api_key="test-key")
    with pytest.raises(UnsupportedParameterError):
        await provider.aupload_file(b"data", **kwargs)
    mock_client.aio.files.upload.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [{"purpose": "batch"}, {"page_size": 1}, {"page_token": "x"}, {"limit": 0}, {"limit": True}, {"cursor": ""}],
)
async def test_invalid_list_options_fail_before_network(kwargs: dict[str, Any]) -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid options reached the provider"))
    try:
        with pytest.raises(AnyLLMError, match=r"(not supported|limit|cursor)"):
            await provider.alist_files(**kwargs)
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_missing_file_has_file_specific_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(
        lambda _: httpx.Response(404, json={"error": {"code": 404, "message": "File not found", "status": "NOT_FOUND"}})
    )
    try:
        with pytest.raises(ProviderFileNotFoundError, match="File not found") as error:
            await provider.aretrieve_file("files/missing")
        assert error.value.status_code == 404
    finally:
        await close_provider(provider)


MISSING_FILE_403 = {
    "error": {
        "code": 403,
        "message": "You do not have permission to access the File doesnotexist123 or it may not exist.",
        "status": "PERMISSION_DENIED",
    }
}


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["retrieve", "delete"])
async def test_unknown_file_403_is_a_missing_file(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(lambda _: httpx.Response(403, json=MISSING_FILE_403))
    try:
        with pytest.raises(ProviderFileNotFoundError, match="may not exist") as error:
            if operation == "retrieve":
                await provider.aretrieve_file("files/doesnotexist123")
            else:
                await provider.adelete_file("files/doesnotexist123")
        assert error.value.status_code == 403
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_request", ["metadata", "content"])
async def test_download_unknown_file_403_is_a_missing_file(
    failing_request: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown ID normally fails the metadata lookup; content covers a file deleted in between."""
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        is_content = ":download" in request.url.path
        if is_content == (failing_request == "content"):
            return httpx.Response(403, json=MISSING_FILE_403)
        return httpx.Response(200, json=GENERATED)

    provider = provider_for(handle)
    try:
        with pytest.raises(ProviderFileNotFoundError, match="may not exist") as error:
            async with provider.adownload_file("files/gen-1"):
                pytest.fail("Missing generated file entered the consumer context")
        assert error.value.status_code == 403
        if failing_request == "metadata":
            assert all(":download" not in request.url.path for request in requests)
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_permission_403_without_missing_file_phrase_stays_authentication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(
        lambda _: httpx.Response(
            403,
            json={"error": {"code": 403, "message": "Permission denied", "status": "PERMISSION_DENIED"}},
        )
    )
    try:
        with pytest.raises(AuthenticationError):
            await provider.aretrieve_file("files/abc-123")
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["upload", "list"])
async def test_collection_404_is_not_a_missing_file(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(
        lambda _: httpx.Response(404, json={"error": {"code": 404, "message": "Wrong endpoint", "status": "NOT_FOUND"}})
    )
    try:
        with pytest.raises(AnyLLMError) as raised:
            if operation == "upload":
                await provider.aupload_file(b"data")
            else:
                await provider.alist_files()
        assert not isinstance(raised.value, ProviderFileNotFoundError)
        assert raised.value.status_code == 404
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("unified", ["0", "1"])
async def test_unreadable_upload_path_is_a_request_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unified: str
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", unified)
    provider = provider_for(lambda _: pytest.fail("Unreadable path reached network"))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(InvalidRequestError, match=r"missing\.pdf") as error:
                await provider.aupload_file(tmp_path / "missing.pdf")
        assert isinstance(error.value.original_exception, FileNotFoundError)
        assert [str(entry.message) for entry in caught] == []
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"extra_headers": "no"}, "extra_headers"),
        ({"max_retries": True}, "max_retries"),
        ({"max_retries": -1}, "max_retries"),
    ],
)
async def test_invalid_retrieve_options_fail_before_network(kwargs: dict[str, Any], match: str) -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid retrieve options reached the network"))
    try:
        with pytest.raises(InvalidRequestError, match=match):
            await provider.aretrieve_file("files/abc-123", **kwargs)
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_id",
    [
        "",
        ".",
        "..",
        "../models",
        "file/../../x",
        "file\\x",
        " file_abc",
        "file_abc\n",
        "abc:download",
        "files/abc:download",
    ],
)
async def test_invalid_file_ids_are_rejected(file_id: str) -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid file ID reached network"))
    try:
        with pytest.raises(InvalidRequestError, match="file ID"):
            await provider.aretrieve_file(file_id)
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_files_prefix_is_accepted_and_bare_ids_are_canonicalized() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=UPLOADED)

    provider = provider_for(handle)
    try:
        await provider.aretrieve_file("files/abc-123")
        await provider.aretrieve_file("abc-123")
        assert [request.url.path for request in requests] == ["/v1beta/files/abc-123", "/v1beta/files/abc-123"]
    finally:
        await close_provider(provider)


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


def generated_handler(stream: CountingStream) -> Callable[[httpx.Request], httpx.Response]:
    def handle(request: httpx.Request) -> httpx.Response:
        if ":download" in request.url.path:
            return httpx.Response(200, headers={"content-type": "text/plain", "content-length": "12"}, stream=stream)
        return httpx.Response(200, json=GENERATED)

    return handle


@pytest.mark.asyncio
@pytest.mark.parametrize("reads", [0, 1, 3])
async def test_download_exposes_headers_without_prefetch_and_closes(reads: int) -> None:
    stream = CountingStream()
    provider = provider_for(generated_handler(stream))
    try:
        async with provider.adownload_file("files/gen-1", chunk_size=4) as download:
            assert isinstance(download, AsyncFileDownload)
            assert download.status_code == 200
            assert download.headers["content-type"] == "text/plain"
            assert stream.reads == 0
            for _ in range(reads):
                assert await anext(download) == b"data"
            assert stream.reads == reads
        assert stream.closed
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_uploaded_file_download_is_rejected_before_content_request() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if ":download" in request.url.path:
            pytest.fail("Uploaded files must not hit the download endpoint")
        return httpx.Response(200, json=UPLOADED)

    provider = provider_for(handle)
    try:
        with pytest.raises(InvalidRequestError, match="cannot be downloaded"):
            async with provider.adownload_file("files/abc-123"):
                pytest.fail("Rejected download entered the consumer context")
        assert len(requests) == 1
        assert ":download" not in requests[0].url.path
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_download_http_error_raises_on_context_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")

    def handle(request: httpx.Request) -> httpx.Response:
        if ":download" in request.url.path:
            return httpx.Response(
                404, json={"error": {"code": 404, "message": "File not found", "status": "NOT_FOUND"}}
            )
        return httpx.Response(200, json=GENERATED)

    provider = provider_for(handle)
    try:
        with pytest.raises(ProviderFileNotFoundError):
            async with provider.adownload_file("files/gen-1"):
                pytest.fail("Failed download entered the consumer context")
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_download_cancellation_closes_upstream() -> None:
    stream = CountingStream()
    provider = provider_for(generated_handler(stream))
    try:
        with pytest.raises(asyncio.CancelledError):
            async with provider.adownload_file("files/gen-1", chunk_size=4) as chunks:
                await anext(chunks)
                raise asyncio.CancelledError
        assert stream.closed
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_stream_failure_is_unified_and_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    stream = CountingStream(fail=True)
    provider = provider_for(generated_handler(stream))
    try:
        with pytest.raises(ProviderError, match="connection lost"):
            async with provider.adownload_file("files/gen-1", chunk_size=4) as download:
                async for _chunk in download:
                    pass
        assert stream.closed
    finally:
        await close_provider(provider)


def test_sync_operations_use_the_same_contract() -> None:
    stream = CountingStream()

    def handle(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        if ":download" in request.url.path:
            return httpx.Response(200, stream=stream)
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json={"files": [UPLOADED]})
        if request.url.path.endswith("/files/gen-1"):
            return httpx.Response(200, json=GENERATED)
        return httpx.Response(200, json=UPLOADED)

    provider = provider_for(handle)
    try:
        assert provider.retrieve_file("files/abc-123").size_bytes == 4
        assert provider.list_files(limit=1).next_cursor is None
        assert provider.delete_file("abc-123").id == "files/abc-123"
        with provider.download_file("files/gen-1", chunk_size=4) as download:
            assert isinstance(download, FileDownload)
            assert download.status_code == 200
            assert stream.reads == 0
            assert next(download) == b"data"
            assert stream.reads == 1
        assert stream.closed
    finally:
        run_async_in_sync(provider.client.aio.aclose())


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [0, -1, None, "64", True, 1.5])
async def test_invalid_download_chunk_size(chunk_size: Any) -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid chunk size reached network"))
    try:
        with pytest.raises(InvalidRequestError, match="chunk_size"):
            async with provider.adownload_file("files/gen-1", chunk_size=chunk_size):
                pass
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["retrieve", "delete", "download"])
async def test_unknown_options_rejected(operation: str) -> None:
    provider = provider_for(lambda _: pytest.fail("Unknown option reached network"))
    try:
        with pytest.raises(UnsupportedParameterError, match="unknown"):
            if operation == "retrieve":
                await provider.aretrieve_file("files/abc-123", unknown=True)
            elif operation == "delete":
                await provider.adelete_file("files/abc-123", unknown=True)
            else:
                async with provider.adownload_file("files/gen-1", unknown=True):
                    pass
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_upload_does_not_retry_by_default(mock_client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    calls = 0

    async def upload(*, file: Any, config: Any = None) -> types.File:
        nonlocal calls
        calls += 1
        raise ClientError(
            code=429,
            response_json={"error": {"message": "Rate limit exceeded", "status": "RESOURCE_EXHAUSTED"}},
        )

    mock_client.aio.files.upload = upload
    provider = GeminiProvider(api_key="test-key")
    with pytest.raises(RateLimitError):
        await provider.aupload_file(b"data")
    assert calls == 1


@pytest.mark.asyncio
async def test_timeout_and_extra_headers_are_forwarded_on_retrieve() -> None:
    requests: list[httpx.Request] = []
    headers = {"X-Test": "yes"}

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=UPLOADED)

    provider = provider_for(handle)
    try:
        await provider.aretrieve_file("files/abc-123", timeout=0.5, extra_headers=headers)
        assert requests[0].extensions["timeout"]["read"] == 0.5
        assert requests[0].headers["x-test"] == "yes"
        assert headers == {"X-Test": "yes"}
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_value", [0, -1, True, "1"])
async def test_invalid_timeout_fails_before_network(timeout_value: Any) -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid timeout reached network"))
    try:
        with pytest.raises(InvalidRequestError, match="timeout"):
            await provider.aretrieve_file("files/abc-123", timeout=timeout_value)
    finally:
        await close_provider(provider)


@pytest.mark.asyncio
async def test_missing_name_is_a_provider_error(mock_client: Any) -> None:
    mock_client.aio.files.get = AsyncMock(return_value=types.File())
    provider = GeminiProvider(api_key="test-key")
    with pytest.raises(ProviderError, match="resource name"):
        await provider.aretrieve_file("files/abc-123")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "mime_type", "expected"),
    [
        ("report.pdf", None, "application/pdf"),
        ("report.pdf", "text/plain", "text/plain"),
        ("no-extension", None, "application/octet-stream"),
    ],
)
async def test_path_upload_guesses_mime_type_and_keeps_an_explicit_one(
    name: str, mime_type: str | None, expected: str, tmp_path: Path, mock_client: Any
) -> None:
    path = tmp_path / name
    path.write_bytes(b"%PDF-1.4")
    captured: list[Any] = []

    async def upload(*, file: Any, config: Any = None) -> types.File:
        captured.append(config)
        return uploaded_file()

    mock_client.aio.files.upload = upload
    provider = GeminiProvider(api_key="test-key")
    await provider.aupload_file(path, mime_type=mime_type)
    assert captured[0].mime_type == expected


@pytest.mark.asyncio
async def test_transport_oserror_is_not_reported_as_an_unreadable_path(
    tmp_path: Path, mock_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the local open is a path error; aiohttp's ClientOSError is an OSError too."""
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    path = tmp_path / "input.csv"
    path.write_bytes(b"a,b\n")

    async def upload(*, file: Any, config: Any = None) -> types.File:
        message = "Connection reset by peer"
        raise ConnectionResetError(message)

    mock_client.aio.files.upload = upload
    provider = GeminiProvider(api_key="test-key")
    with pytest.raises(AnyLLMError) as error:
        await provider.aupload_file(path)
    assert "Cannot open upload path" not in str(error.value)
    assert isinstance(error.value.original_exception, ConnectionResetError)


@pytest.mark.asyncio
async def test_upload_closes_handles_it_opens_but_not_the_callers(tmp_path: Path, mock_client: Any) -> None:
    path = tmp_path / "input.csv"
    path.write_bytes(b"a,b\n")
    seen: list[Any] = []

    async def upload(*, file: Any, config: Any = None) -> types.File:
        seen.append(file)
        return uploaded_file()

    mock_client.aio.files.upload = upload
    provider = GeminiProvider(api_key="test-key")
    caller_handle = BytesIO(b"a,b\n")
    await provider.aupload_file(path)
    await provider.aupload_file(b"a,b\n")
    await provider.aupload_file(caller_handle)
    assert seen[0].closed
    assert seen[1].closed
    assert not caller_handle.closed
    caller_handle.close()


def test_sub_millisecond_timeout_rounds_up_to_one_millisecond() -> None:
    from any_llm.providers.gemini.files import file_http_options

    options = file_http_options({"timeout": 0.0004})
    assert options is not None
    assert options.timeout == 1

    options = file_http_options({"timeout": 1.0001})
    assert options is not None
    assert options.timeout == 1001


@pytest.mark.asyncio
async def test_delete_error_other_than_a_missing_file_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(
        lambda _: httpx.Response(429, json={"error": {"code": 429, "message": "Quota", "status": "RESOURCE_EXHAUSTED"}})
    )
    try:
        with pytest.raises(RateLimitError):
            await provider.adelete_file("files/abc-123")
    finally:
        await close_provider(provider)


def test_http_status_is_unknown_without_a_usable_response() -> None:
    from any_llm.providers.gemini.files import _http_status

    assert _http_status(ClientError(403, {"error": {"message": "x"}})) is None
    no_status = ClientError(403, {"error": {"message": "x"}}, response=httpx.Response(403))
    no_status.response = type("Response", (), {"status_code": True})()
    assert _http_status(no_status) is None


@pytest.mark.asyncio
async def test_download_without_an_httpx_client_is_a_provider_error() -> None:
    provider = provider_for(lambda _: pytest.fail("Download reached the network without an HTTP client"))
    generated = types.File.model_validate(GENERATED)
    try:
        with (
            patch.object(provider.client.aio.files, "get", AsyncMock(return_value=generated)),
            patch.object(provider.client._api_client, "_async_httpx_client", None),
        ):
            with pytest.raises(ProviderError, match="async HTTP client"):
                async with provider.adownload_file("files/gen-1"):
                    pytest.fail("Download opened without an HTTP client")
    finally:
        await close_provider(provider)


def test_default_client_keeps_an_httpx_client_for_downloads_beside_aiohttp() -> None:
    """In a default install metadata goes over aiohttp while downloads use the SDK's httpx client.

    Every other unit test injects an httpx transport, which opts the SDK out of aiohttp, so
    this pins the one assumption the download path makes about the default configuration.
    """
    pytest.importorskip("aiohttp")
    provider = GeminiProvider(api_key="test-key")
    api_client = provider.client._api_client
    assert api_client._use_aiohttp()
    assert api_client._async_httpx_client is not None


@pytest.mark.asyncio
async def test_download_metadata_error_other_than_a_missing_file_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(
        lambda _: httpx.Response(429, json={"error": {"code": 429, "message": "Quota", "status": "RESOURCE_EXHAUSTED"}})
    )
    try:
        with pytest.raises(RateLimitError):
            async with provider.adownload_file("files/gen-1"):
                pytest.fail("A failed metadata lookup entered the consumer context")
    finally:
        await close_provider(provider)
