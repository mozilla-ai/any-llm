import asyncio
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

import pytest

from any_llm.exceptions import InvalidRequestError, MissingApiKeyError
from any_llm.providers.gemini.gemini import GeminiProvider
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
    try:
        await provider.client.aio.aclose()
    except Exception as exc:
        errors.append(exc)
    # Uploads run on the sync client, which aio.aclose() does not close.
    try:
        provider.client.close()
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


@pytest.mark.asyncio
async def test_gemini_uploaded_file_is_usable_in_a_completion(tmp_path: Path) -> None:
    """A path upload guesses its MIME type, and filename lets the file URI carry one into a request."""
    try:
        provider = GeminiProvider()
    except MissingApiKeyError:
        if "gemini" in EXPECTED_PROVIDERS:
            raise
        pytest.skip("GEMINI_API_KEY/GOOGLE_API_KEY is not configured")
    file_id: str | None = None
    try:
        path = tmp_path / "codes.txt"
        path.write_bytes(b"The access code is 8391.\n")
        uploaded = await provider.aupload_file(path)
        file_id = uploaded.id
        assert uploaded.mime_type == "text/plain"

        deadline = asyncio.get_running_loop().time() + READY_TIMEOUT_SECONDS
        while uploaded.status == "PROCESSING":
            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail(f"File {file_id} stayed PROCESSING for {READY_TIMEOUT_SECONDS}s")
            await asyncio.sleep(1)
            uploaded = await provider.aretrieve_file(file_id)
        assert uploaded.status == "ACTIVE", uploaded.model_dump()

        uri = (uploaded.model_extra or {}).get("uri")
        assert isinstance(uri, str)
        response = await provider.acompletion(
            model="gemini-3-flash-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "file", "file": {"file_data": uri, "filename": uploaded.filename}},
                        {"type": "text", "text": "What is the access code? Answer with digits only."},
                    ],
                }
            ],
        )
        assert "8391" in (response.choices[0].message.content or "")
    finally:
        await cleanup_files(provider, [file_id] if file_id is not None else [])
