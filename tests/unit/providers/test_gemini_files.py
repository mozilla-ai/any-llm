import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import httpx
import pytest
from google.genai import errors, types

from any_llm import AnyLLM
from any_llm.exceptions import (
    AnyLLMError,
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    ProviderFileNotFoundError,
    UnsupportedParameterError,
)
from any_llm.providers.gemini import GeminiProvider
from any_llm.providers.gemini.files import convert_metadata

META = {
    "name": "files/abc123",
    "displayName": "input.csv",
    "mimeType": "text/csv",
    "sizeBytes": "4",
    "createTime": "2026-09-14T12:00:00Z",
    "expirationTime": "2026-09-16T12:00:00Z",
    "state": "ACTIVE",
    "source": "UPLOADED",
    "uri": "https://generativelanguage.googleapis.com/v1beta/files/abc123",
}


def provider_for(handler: Callable[[httpx.Request], httpx.Response]) -> GeminiProvider:
    return GeminiProvider(
        api_key="test-key",
        http_options=types.HttpOptions(httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))),
    )


def upload_handler(
    requests: list[httpx.Request], file_json: dict[str, object] | None = None
) -> Callable[[httpx.Request], httpx.Response]:
    """Answer the resumable upload handshake: a session URL, then the finalized file."""

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.headers.get("x-goog-upload-command") == "start":
            return httpx.Response(200, json={}, headers={"x-goog-upload-url": "https://upload.test/session"})
        return httpx.Response(
            200,
            json={"file": file_json if file_json is not None else META},
            headers={"x-goog-upload-status": "final"},
        )

    return handle


@pytest.mark.asyncio
async def test_upload_path_defaults_filename_and_mime_type(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    path.write_bytes(b"a,b\n")
    requests: list[httpx.Request] = []
    provider = provider_for(upload_handler(requests))

    result = await provider.aupload_file(path)

    assert result.id == "files/abc123"
    assert result.filename == "input.csv"
    assert result.size_bytes == 4
    assert result.mime_type == "text/csv"
    assert result.created_at == datetime(2026, 9, 14, 12, 0, tzinfo=UTC)
    assert result.expires_at == datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
    assert result.status == "ACTIVE"
    assert result.downloadable is False
    assert result.model_extra == {"source": "UPLOADED", "uri": META["uri"]}
    assert requests[0].url.path == "/upload/v1beta/files"
    assert requests[0].headers["x-goog-upload-header-content-type"] == "text/csv"
    assert b'"display_name": "input.csv"' in requests[0].content
    assert requests[1].content == b"a,b\n"


@pytest.mark.asyncio
async def test_upload_accepts_bytes_and_handles_with_explicit_metadata() -> None:
    requests: list[httpx.Request] = []
    provider = provider_for(upload_handler(requests))

    await provider.aupload_file(b"a,b\n", filename="bytes.csv", mime_type="text/csv")
    await provider.aupload_file(BytesIO(b"a,b\n"), filename="handle.csv", mime_type="text/csv")

    assert b'"display_name": "bytes.csv"' in requests[0].content
    assert b'"display_name": "handle.csv"' in requests[2].content
    assert requests[1].content == requests[3].content == b"a,b\n"


@pytest.mark.asyncio
async def test_upload_without_a_mime_type_falls_back_to_octet_stream() -> None:
    requests: list[httpx.Request] = []
    provider = provider_for(upload_handler(requests))

    await provider.aupload_file(b"data")

    assert requests[0].headers["x-goog-upload-header-content-type"] == "application/octet-stream"


@pytest.mark.asyncio
async def test_upload_does_not_retry_by_default_and_honors_an_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    attempts: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    provider = provider_for(handle)
    with pytest.raises(ProviderError, match="unavailable"):
        await provider.aupload_file(b"data")
    assert len(attempts) == 1

    attempts.clear()
    with pytest.raises(ProviderError, match="unavailable"):
        await provider.aupload_file(b"data", max_retries=1)
    assert len(attempts) == 2


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_options() -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid options reached the provider"))

    with pytest.raises(UnsupportedParameterError, match="purpose"):
        await provider.aupload_file(b"data", purpose="user_data")
    with pytest.raises(UnsupportedParameterError, match="expires_in"):
        await provider.aupload_file(b"data", expires_in=3600)
    with pytest.raises(UnsupportedParameterError, match="extra_headers"):
        await provider.aupload_file(b"data", extra_headers={"x-custom": "value"})
    with pytest.raises(UnsupportedParameterError, match="order"):
        await provider.aupload_file(b"data", order="asc")
    with pytest.raises(InvalidRequestError, match="max_retries"):
        await provider.aupload_file(b"data", max_retries=-1)
    with pytest.raises(InvalidRequestError, match="timeout"):
        await provider.aupload_file(b"data", timeout=0)


@pytest.mark.asyncio
async def test_upload_reports_an_unopenable_path_as_a_caller_error(tmp_path: Path) -> None:
    provider = provider_for(lambda _: pytest.fail("A missing path reached the provider"))
    missing = tmp_path / "absent.csv"

    with pytest.raises(InvalidRequestError, match="Cannot open upload path") as error:
        await provider.aupload_file(missing)
    assert isinstance(error.value.original_exception, OSError)


@pytest.mark.asyncio
async def test_list_maps_pagination_and_fetches_one_page() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"files": [META], "nextPageToken": "token-2"})

    provider = provider_for(handle)
    page = await provider.alist_files(limit=1, cursor="token-1")

    assert [item.id for item in page.data] == ["files/abc123"]
    assert page.next_cursor == "token-2"
    assert len(requests) == 1
    assert requests[0].url.params["pageSize"] == "1"
    assert requests[0].url.params["pageToken"] == "token-1"


@pytest.mark.asyncio
async def test_list_final_page_has_no_cursor_and_tolerates_an_empty_body() -> None:
    provider = provider_for(lambda _: httpx.Response(200, json={}))

    page = await provider.alist_files()

    assert page.data == []
    assert page.next_cursor is None


@pytest.mark.asyncio
async def test_list_rejects_unsupported_options() -> None:
    provider = provider_for(lambda _: pytest.fail("Invalid options reached the provider"))

    with pytest.raises(UnsupportedParameterError, match="purpose"):
        await provider.alist_files(purpose="batch")
    with pytest.raises(UnsupportedParameterError, match="order"):
        await provider.alist_files(order="asc")
    with pytest.raises(InvalidRequestError, match="limit"):
        await provider.alist_files(limit=0)
    with pytest.raises(InvalidRequestError, match="limit"):
        await provider.alist_files(limit=True)
    with pytest.raises(InvalidRequestError, match="cursor"):
        await provider.alist_files(cursor="  ")


@pytest.mark.asyncio
async def test_retrieve_leaves_missing_metadata_unknown_and_sends_request_options() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"name": "files/abc123", "state": "PROCESSING"})

    provider = provider_for(handle)
    result = await provider.aretrieve_file("files/abc123", extra_headers={"x-custom": "value"}, timeout=30)

    assert result.id == "files/abc123"
    assert result.status == "PROCESSING"
    assert result.filename is None
    assert result.size_bytes is None
    assert result.downloadable is None
    assert requests[0].url.path == "/v1beta/files/abc123"
    assert requests[0].headers["x-custom"] == "value"


@pytest.mark.asyncio
async def test_delete_acknowledges_the_deleted_id() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    provider = provider_for(handle)
    result = await provider.adelete_file("abc123")

    assert result.id == "abc123"
    assert result.deleted is True
    assert requests[0].method == "DELETE"
    assert requests[0].url.path == "/v1beta/files/abc123"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_id",
    [
        "",
        ".",
        "..",
        "files/",
        "other/abc123",
        "abc 123",
        "files/abc/123",
        "abc.123",
        # A colon would otherwise reach the SDK and address a subresource, as in
        # files/abc:download?alt=media.
        "abc123:download",
        "files/abc123:download",
        "https://generativelanguage.googleapis.com/v1beta/files/abc123",
    ],
)
async def test_file_id_validation_rejects_names_outside_the_resource_form(file_id: str) -> None:
    provider = provider_for(lambda _: pytest.fail("An invalid file ID reached the provider"))

    with pytest.raises(InvalidRequestError, match="file ID"):
        await provider.aretrieve_file(file_id)
    with pytest.raises(InvalidRequestError, match="file ID"):
        await provider.adelete_file(file_id)


@pytest.mark.asyncio
async def test_missing_file_has_file_specific_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini hides whether a file exists behind 403 PERMISSION_DENIED instead of 404."""
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(
        lambda _: httpx.Response(
            403,
            json={
                "error": {
                    "code": 403,
                    "message": "You do not have permission to access the File missing or it may not exist.",
                    "status": "PERMISSION_DENIED",
                }
            },
        )
    )

    with pytest.raises(ProviderFileNotFoundError, match="may not exist") as error:
        await provider.aretrieve_file("files/missing")
    assert error.value.status_code == 403

    with pytest.raises(ProviderFileNotFoundError, match="may not exist"):
        await provider.adelete_file("files/missing")


@pytest.mark.asyncio
async def test_a_genuine_permission_failure_is_not_reported_as_a_missing_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(
        lambda _: httpx.Response(
            403,
            json={
                "error": {
                    "code": 403,
                    "message": "Generative Language API has not been used in this project before.",
                    "status": "PERMISSION_DENIED",
                }
            },
        )
    )

    with pytest.raises(AuthenticationError) as error:
        await provider.aretrieve_file("files/abc123")
    assert not isinstance(error.value, ProviderFileNotFoundError)


@pytest.mark.asyncio
async def test_argument_errors_survive_unified_exceptions_without_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = provider_for(lambda _: pytest.fail("Invalid options reached the provider"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(UnsupportedParameterError, match="purpose"):
            await provider.aupload_file(b"data", purpose="user_data")
        with pytest.raises(InvalidRequestError, match="limit"):
            await provider.alist_files(limit=0)
        with pytest.raises(InvalidRequestError, match="file ID"):
            await provider.aretrieve_file("")
    assert [str(entry.message) for entry in caught] == []


def test_metadata_without_a_resource_name_is_a_provider_error() -> None:
    with pytest.raises(ProviderError, match="resource name"):
        convert_metadata(types.File(display_name="input.csv"))


def test_metadata_without_a_state_leaves_the_status_unknown() -> None:
    result = convert_metadata(types.File(name="files/abc123"))

    assert result.status is None
    assert result.downloadable is None


def test_generated_files_leave_downloadability_unknown() -> None:
    result = convert_metadata(types.File.model_validate({**META, "source": "GENERATED"}))

    assert result.downloadable is None
    assert result.model_extra == {"source": "GENERATED", "uri": META["uri"]}


def test_sync_methods_mirror_their_async_counterparts() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.headers.get("x-goog-upload-command") is not None:
            return upload_handler(requests)(request)
        requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(200, json={})
        if request.url.path.endswith("/files"):
            return httpx.Response(200, json={"files": [META]})
        return httpx.Response(200, json=META)

    provider = provider_for(handle)

    assert provider.upload_file(b"a,b\n", mime_type="text/csv").id == "files/abc123"
    assert [item.id for item in provider.list_files().data] == ["files/abc123"]
    assert provider.retrieve_file("files/abc123").id == "files/abc123"
    assert provider.delete_file("files/abc123").deleted is True


def test_file_capabilities_exclude_downloads() -> None:
    metadata = GeminiProvider.get_provider_metadata()

    assert metadata.files is True
    assert set(metadata.file_operations) == {"upload", "list", "retrieve", "delete"}
    assert AnyLLM.get_provider_class("vertexai").get_provider_metadata().files is False


@pytest.mark.asyncio
async def test_download_is_not_supported() -> None:
    provider = provider_for(lambda _: pytest.fail("A download reached the provider"))

    with pytest.raises(NotImplementedError, match="download"):
        async with provider.adownload_file("files/abc123"):
            pass


@pytest.mark.asyncio
async def test_a_missing_file_403_is_left_alone_without_unified_exceptions() -> None:
    """The translation follows the unified-exceptions contract like any other provider failure."""
    provider = provider_for(
        lambda _: httpx.Response(
            403,
            json={
                "error": {
                    "code": 403,
                    "message": "You do not have permission to access the File missing or it may not exist.",
                    "status": "PERMISSION_DENIED",
                }
            },
        )
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(errors.APIError) as error:
            await provider.aretrieve_file("files/missing")
    assert not isinstance(error.value, AnyLLMError)
    assert error.value.code == 403


@pytest.mark.asyncio
async def test_an_instance_flag_enables_the_missing_file_translation() -> None:
    """The per-instance option is honored without the environment variable."""
    provider = GeminiProvider(
        api_key="test-key",
        unified_exceptions=True,
        http_options=types.HttpOptions(
            httpx_async_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(
                        403,
                        json={
                            "error": {
                                "code": 403,
                                "message": "You do not have permission to access the File x or it may not exist.",
                                "status": "PERMISSION_DENIED",
                            }
                        },
                    )
                )
            )
        ),
    )

    with pytest.raises(ProviderFileNotFoundError) as error:
        await provider.adelete_file("files/x")
    assert error.value.status_code == 403
    assert error.value.original_exception is not None


@pytest.mark.asyncio
async def test_max_retries_does_not_reach_the_resumable_byte_transfer() -> None:
    """The SDK retries upload chunks internally, resuming the same session rather than a new one."""
    attempts = {"start": 0, "body": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.headers.get("x-goog-upload-command") == "start":
            attempts["start"] += 1
            return httpx.Response(200, json={}, headers={"x-goog-upload-url": "https://upload.test/session"})
        attempts["body"] += 1
        return httpx.Response(503, json={"error": {"message": "unavailable"}})

    provider = provider_for(handle)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(errors.APIError):
            await provider.aupload_file(b"data", mime_type="text/plain")

    assert attempts["start"] == 1
    assert attempts["body"] > 1
