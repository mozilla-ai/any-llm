import asyncio
import sys
import uuid
from collections.abc import Iterable

import pytest

from any_llm.exceptions import MissingApiKeyError, ProviderFileNotFoundError
from any_llm.providers.gemini import GeminiProvider
from tests.constants import EXPECTED_PROVIDERS

READY_TIMEOUT_SECONDS = 60


async def cleanup_files(provider: GeminiProvider, file_ids: Iterable[str]) -> None:
    primary_error = sys.exc_info()[1]
    errors: list[Exception] = []
    for file_id in file_ids:
        try:
            await provider.adelete_file(file_id)
        except Exception as exc:
            errors.append(exc)
    if errors:
        if primary_error is not None:
            for error in errors:
                primary_error.add_note(f"Files cleanup failed: {error!r}")
        else:
            message = "Files cleanup failed"
            raise ExceptionGroup(message, errors)


def build_provider(*, unified_exceptions: bool = False) -> GeminiProvider:
    try:
        return GeminiProvider(unified_exceptions=unified_exceptions)
    except MissingApiKeyError:
        if "gemini" in EXPECTED_PROVIDERS:
            raise
        pytest.skip("GEMINI_API_KEY/GOOGLE_API_KEY is not configured")


async def wait_until_active(provider: GeminiProvider, file_id: str) -> None:
    deadline = asyncio.get_running_loop().time() + READY_TIMEOUT_SECONDS
    while True:
        metadata = await provider.aretrieve_file(file_id)
        if metadata.status == "ACTIVE":
            return
        assert metadata.status == "PROCESSING", f"Upload failed: {metadata.model_dump()}"
        if asyncio.get_running_loop().time() >= deadline:
            pytest.fail(f"File {file_id} stayed in {metadata.status} for {READY_TIMEOUT_SECONDS}s")
        await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_gemini_files_lifecycle() -> None:
    provider = build_provider()
    file_id: str | None = None
    try:
        uploaded = await provider.aupload_file(
            b"name,value\nexample,1\n",
            filename=f"any-llm-test-{uuid.uuid4().hex}.csv",
            mime_type="text/csv",
        )
        file_id = uploaded.id
        assert file_id.startswith("files/")
        assert uploaded.size_bytes == 21
        assert uploaded.downloadable is False
        assert uploaded.status in {"PROCESSING", "ACTIVE"}

        retrieved = await provider.aretrieve_file(file_id)
        assert retrieved.id == file_id
        assert retrieved.mime_type == "text/csv"
        assert retrieved.expires_at is not None

        first_page = await provider.alist_files(limit=1)
        assert len(first_page.data) <= 1
        if first_page.next_cursor is not None:
            next_page = await provider.alist_files(limit=1, cursor=first_page.next_cursor)
            assert len(next_page.data) <= 1
            assert {item.id for item in next_page.data}.isdisjoint({item.id for item in first_page.data})

        deleted = await provider.adelete_file(file_id)
        deleted_file_id = file_id
        file_id = None
        assert deleted.id == deleted_file_id
        assert deleted.deleted is True
    finally:
        await cleanup_files(provider, [file_id] if file_id is not None else [])


@pytest.mark.asyncio
async def test_gemini_uploaded_file_is_usable_in_a_completion() -> None:
    provider = build_provider()
    file_id: str | None = None
    try:
        uploaded = await provider.aupload_file(
            b"The access code is 8391.\n",
            filename=f"any-llm-test-{uuid.uuid4().hex}.txt",
            mime_type="text/plain",
        )
        file_id = uploaded.id
        await wait_until_active(provider, file_id)
        # The provider file URI is native metadata, so it stays in the extras rather than
        # becoming a shared FileMetadata field.
        uri = (uploaded.model_extra or {}).get("uri")
        assert isinstance(uri, str)

        response = await provider.acompletion(
            model="gemini-2.5-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "file", "file": {"file_data": uri, "filename": "codes.txt"}},
                        {"type": "text", "text": "What is the access code? Answer with digits only."},
                    ],
                }
            ],
        )
        assert "8391" in (response.choices[0].message.content or "")
    finally:
        await cleanup_files(provider, [file_id] if file_id is not None else [])


@pytest.mark.asyncio
async def test_gemini_reports_an_unusable_file_id_as_a_missing_file() -> None:
    """Gemini answers a file-scoped call for an unknown ID with 403, not 404."""
    provider = build_provider(unified_exceptions=True)

    with pytest.raises(ProviderFileNotFoundError) as error:
        await provider.aretrieve_file(f"files/{uuid.uuid4().hex[:12]}")
    assert error.value.status_code == 403

    with pytest.raises(ProviderFileNotFoundError):
        await provider.adelete_file(f"files/{uuid.uuid4().hex[:12]}")
