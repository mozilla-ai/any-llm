import asyncio
from collections.abc import AsyncIterator
from contextlib import nullcontext
from typing import Any

import httpx
import pytest
from openai import APIStatusError
from typing_extensions import override

from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderFileNotFoundError,
    RateLimitError,
)
from any_llm.providers.azureopenai.azureopenai import AzureopenaiProvider
from any_llm.types.files import FileDeleted, FileMetadata, FilePage
from any_llm.utils.aio import run_async_in_sync

META: dict[str, Any] = {
    "id": "file-azure",
    "object": "file",
    "filename": "input.jsonl",
    "bytes": 3,
    "created_at": 1700000000,
    "expires_at": 1700003600,
    "purpose": "batch",
    "status": "processed",
}
OPERATIONS = ("upload", "list", "retrieve", "delete", "download")


async def call_operation(provider: AzureopenaiProvider, operation: str, **kwargs: Any) -> Any:
    if operation == "upload":
        return await provider.aupload_file(b"{}\n", purpose="batch", **kwargs)
    if operation == "list":
        return await provider.alist_files(**kwargs)
    if operation == "retrieve":
        return await provider.aretrieve_file("file-azure", **kwargs)
    if operation == "delete":
        return await provider.adelete_file("file-azure", **kwargs)
    async with provider.adownload_file("file-azure", **kwargs) as download:
        return b"".join([chunk async for chunk in download])


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("credential", ["api_key", "azure_ad_token", "sync_token", "async_token"])
async def test_azure_files_use_v1_routes_and_azure_credentials(operation: str, credential: str) -> None:
    requests: list[httpx.Request] = []

    async def async_token() -> str:
        return "azure-secret"

    credentials: dict[str, Any] = (
        {"azure_ad_token_provider": async_token if credential == "async_token" else lambda: "azure-secret"}
        if credential in {"sync_token", "async_token"}
        else {credential: "azure-secret"}
    )

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if operation == "list":
            return httpx.Response(200, json={"data": [META], "has_more": False})
        if operation == "delete":
            return httpx.Response(200, json={"id": "file-azure", "object": "file", "deleted": True})
        if operation == "download":
            return httpx.Response(200, content=b"{}\n")
        return httpx.Response(200, json=META)

    provider = AzureopenaiProvider(
        api_base="https://resource.openai.azure.com",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        **credentials,
    )
    async with provider.client:
        result = await call_operation(provider, operation, timeout=7, extra_headers={"x-test": "yes"})
    assert len(requests) == 1
    request = requests[0]
    suffix = {
        "upload": "",
        "list": "",
        "retrieve": "/file-azure",
        "delete": "/file-azure",
        "download": "/file-azure/content",
    }[operation]
    assert request.url.path == f"/openai/v1/files{suffix}"
    assert request.url.host == "resource.openai.azure.com"
    assert request.method == {"upload": "POST", "delete": "DELETE"}.get(operation, "GET")
    assert not request.url.params
    assert request.headers["authorization"] == "Bearer azure-secret"
    assert request.headers["x-test"] == "yes"
    assert request.extensions["timeout"]["read"] == 7
    if operation in {"upload", "retrieve"}:
        assert isinstance(result, FileMetadata)
        assert result.size_bytes == 3
        assert result.purpose == "batch"
        assert result.status == "processed"
        assert result.created_at is not None
        assert result.expires_at is not None
        assert (result.expires_at - result.created_at).total_seconds() == 3600
    elif operation == "list":
        assert isinstance(result, FilePage)
        assert result.next_cursor is None
        assert result.data[0].id == "file-azure"
    elif operation == "delete":
        assert isinstance(result, FileDeleted)
        assert result.deleted is True
    else:
        assert result == b"{}\n"
    if operation == "upload":
        assert b'name="purpose"\r\n\r\nbatch' in request.content
        assert b"expires_after" not in request.content


@pytest.mark.asyncio
async def test_azure_file_expiry_and_pagination_use_shared_parameters() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json=META)
        if "after" in request.url.params:
            return httpx.Response(200, json={"data": [], "has_more": False})
        return httpx.Response(200, json={"data": [META], "has_more": True})

    provider = AzureopenaiProvider(
        api_base="https://resource.openai.azure.com/openai/v1/",
        api_key="key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    async with provider.client:
        await provider.aupload_file(b"{}\n", purpose="batch", expires_in=3600)
        page = await provider.alist_files(limit=1, purpose="batch", order="asc")
        assert page.next_cursor == "file-azure"
        last = await provider.alist_files(limit=1, cursor=page.next_cursor, purpose="batch", order="asc")
    assert b'name="expires_after[anchor]"\r\n\r\ncreated_at' in requests[0].content
    assert b'name="expires_after[seconds]"\r\n\r\n3600' in requests[0].content
    assert requests[1].url.params == httpx.QueryParams(limit=1, purpose="batch", order="asc")
    assert requests[2].url.params == httpx.QueryParams(limit=1, purpose="batch", order="asc", after="file-azure")
    assert last.data == []
    assert last.next_cursor is None


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("status", [400, 401, 404, 429])
@pytest.mark.parametrize("unified", [False, True])
async def test_azure_files_error_mapping(
    operation: str, status: int, unified: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1" if unified else "0")
    provider = AzureopenaiProvider(
        api_base="https://resource.openai.azure.com",
        api_key="key",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    status, headers={"Retry-After": "30"}, json={"error": {"message": "request rejected"}}
                )
            )
        ),
    )
    missing_error = ProviderFileNotFoundError if operation in {"retrieve", "delete", "download"} else ModelNotFoundError
    expected = (
        {400: InvalidRequestError, 401: AuthenticationError, 404: missing_error, 429: RateLimitError}[status]
        if unified
        else APIStatusError
    )
    async with provider.client:
        with pytest.raises(expected) as error:
            await call_operation(provider, operation)
    if unified:
        assert isinstance(error.value, AnyLLMError)
        assert error.value.provider_name == "azureopenai"
        assert error.value.status_code == status
        if isinstance(error.value, RateLimitError):
            assert error.value.retry_after == "30"


@pytest.mark.asyncio
@pytest.mark.parametrize("retries", [None, 0, 1])
async def test_azure_file_upload_retry_override_refreshes_entra_token(retries: int | None) -> None:
    requests: list[httpx.Request] = []
    tokens = iter(["first-token", "second-token"])

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "rate limit"}})

    provider = AzureopenaiProvider(
        api_base="https://resource.openai.azure.com",
        azure_ad_token_provider=lambda: next(tokens),
        max_retries=3,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    options: dict[str, Any] = {} if retries is None else {"max_retries": retries}
    async with provider.client:
        with pytest.raises(APIStatusError):
            await provider.aupload_file(b"{}\n", purpose="batch", **options)
    expected_requests = 1 if retries is None else retries + 1
    assert len(requests) == expected_requests
    assert [request.headers["authorization"] for request in requests] == ["Bearer first-token", "Bearer second-token"][
        :expected_requests
    ]


def test_azure_files_sync_lifecycle_uses_v1_routes() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path.startswith("/openai/v1/files")
        assert request.headers["authorization"] == "Bearer azure-key"
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"{}\n")
        if request.method == "DELETE":
            return httpx.Response(200, json={"id": "file-azure", "object": "file", "deleted": True})
        if request.method == "GET" and request.url.path == "/openai/v1/files":
            return httpx.Response(200, json={"data": [META], "has_more": False})
        return httpx.Response(200, json=META)

    provider = AzureopenaiProvider(
        api_base="https://resource.openai.azure.com",
        api_key="azure-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    try:
        assert provider.upload_file(b"{}\n", purpose="batch").size_bytes == 3
        assert provider.retrieve_file("file-azure").id == "file-azure"
        assert provider.list_files().next_cursor is None
        with provider.download_file("file-azure", chunk_size=1) as download:
            assert download.status_code == 200
            assert list(download) == [b"{", b"}", b"\n"]
        assert provider.delete_file("file-azure").deleted
        assert len(requests) == 5
    finally:
        run_async_in_sync(provider.client.close())


class TrackingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.reads = 0
        self.closed = False

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        for _ in range(3):
            self.reads += 1
            yield b"data"

    @override
    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("consumption", ["unread", "early", "full", "cancelled"])
async def test_azure_file_download_is_lazy_and_closes(consumption: str) -> None:
    stream = TrackingStream()
    provider = AzureopenaiProvider(
        api_base="https://resource.openai.azure.com",
        api_key="key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))),
    )
    async with provider.client:
        with pytest.raises(asyncio.CancelledError) if consumption == "cancelled" else nullcontext():
            async with provider.adownload_file("file-azure", chunk_size=4) as download:
                assert stream.reads == 0
                assert download.status_code == 200
                if consumption in {"early", "cancelled"}:
                    assert await anext(download) == b"data"
                    assert stream.reads == 1
                    if consumption == "cancelled":
                        raise asyncio.CancelledError
                elif consumption == "full":
                    assert b"".join([chunk async for chunk in download]) == b"datadatadata"
        assert stream.closed
