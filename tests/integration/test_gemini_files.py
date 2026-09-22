import sys
import uuid
from collections.abc import Iterable

import pytest

from any_llm.exceptions import InvalidRequestError, MissingApiKeyError
from any_llm.providers.gemini.gemini import GeminiProvider
from tests.constants import EXPECTED_PROVIDERS


async def cleanup_files(provider: GeminiProvider, file_ids: Iterable[str]) -> None:
    primary_error = sys.exc_info()[1]
    errors: list[Exception] = []
    for file_id in file_ids:
        try:
            await provider.adelete_file(file_id)
        except Exception as exc:
            errors.append(exc)
    try:
        await provider.client.aio.aclose()
    except Exception as exc:
        errors.append(exc)
    if errors:
        if primary_error is not None:
            for error in errors:
                primary_error.add_note(f"Files cleanup failed: {error!r}")
        else:
            message = "Files cleanup failed"
            raise ExceptionGroup(message, errors)


@pytest.mark.asyncio
async def test_gemini_files_lifecycle() -> None:
    try:
        provider = GeminiProvider()
    except MissingApiKeyError:
        if "gemini" in EXPECTED_PROVIDERS:
            raise
        pytest.skip("GEMINI_API_KEY or GOOGLE_API_KEY is not configured")
    file_id: str | None = None
    try:
        uploaded = await provider.aupload_file(
            b"name,value\nexample,1\n",
            filename=f"any-llm-test-{uuid.uuid4().hex}.csv",
            mime_type="text/csv",
        )
        file_id = uploaded.id
        assert uploaded.size_bytes == 21
        assert uploaded.downloadable is False
        assert uploaded.status in {None, "PROCESSING", "ACTIVE"}
        retrieved = await provider.aretrieve_file(file_id)
        assert retrieved.id == file_id
        first_page = await provider.alist_files(limit=1)
        assert len(first_page.data) <= 1
        if first_page.next_cursor is not None:
            next_page = await provider.alist_files(limit=1, cursor=first_page.next_cursor)
            assert len(next_page.data) <= 1
        with pytest.raises(InvalidRequestError, match="cannot be downloaded"):
            async with provider.adownload_file(file_id):
                pytest.fail("User-uploaded Gemini files must not start a download")
        deleted = await provider.adelete_file(file_id)
        deleted_file_id = file_id
        file_id = None
        assert deleted.id == deleted_file_id
    finally:
        await cleanup_files(provider, [file_id] if file_id is not None else [])
