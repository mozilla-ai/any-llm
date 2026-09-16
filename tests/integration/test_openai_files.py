import json
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path

import pytest

from any_llm.exceptions import InvalidRequestError, MissingApiKeyError, ProviderFileNotFoundError
from any_llm.providers.openai.openai import OpenaiProvider
from any_llm.utils.aio import run_async_in_sync
from tests.constants import EXPECTED_PROVIDERS

# Uploading a batch input does not submit a batch or invoke a model.
BATCH_CONTENT = (
    json.dumps(
        {
            "custom_id": "files-test",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]},
        }
    )
    + "\n"
).encode()


def create_provider() -> OpenaiProvider:
    try:
        return OpenaiProvider(max_retries=0, timeout=30)
    except MissingApiKeyError:
        if "openai" in EXPECTED_PROVIDERS:
            raise
        pytest.skip("OPENAI_API_KEY is not configured")


async def cleanup_files(
    provider: OpenaiProvider, file_ids: Iterable[str], *, primary_error: BaseException | None = None
) -> None:
    if primary_error is None:
        primary_error = sys.exc_info()[1]
    errors: list[Exception] = []
    for file_id in file_ids:
        try:
            await provider.adelete_file(file_id)
        except Exception as exc:
            errors.append(exc)
    try:
        await provider.client.close()
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
async def test_openai_files_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = create_provider()
    file_ids: list[str] = []
    path = tmp_path / f"any-llm-test-{uuid.uuid4().hex}.jsonl"
    try:
        path.write_bytes(BATCH_CONTENT)
        uploaded = await provider.aupload_file(path, purpose="batch", expires_in=3600)
        file_ids.append(uploaded.id)
        assert uploaded.filename == path.name
        assert uploaded.size_bytes == len(BATCH_CONTENT)
        assert uploaded.purpose == "batch"
        assert uploaded.created_at is not None
        assert uploaded.expires_at is not None
        assert (uploaded.expires_at - uploaded.created_at).total_seconds() == 3600
        retrieved = await provider.aretrieve_file(uploaded.id)
        assert retrieved.id == uploaded.id
        assert retrieved.size_bytes == uploaded.size_bytes

        # Two files exercise the cursor even in an otherwise empty account.
        second = await provider.aupload_file(
            BATCH_CONTENT,
            filename=f"any-llm-test-{uuid.uuid4().hex}.jsonl",
            purpose="batch",
            expires_in=3600,
        )
        file_ids.append(second.id)
        first_page = await provider.alist_files(limit=1, purpose="batch", order="desc")
        assert len(first_page.data) == 1
        assert first_page.data[0].purpose == "batch"
        assert first_page.next_cursor == first_page.data[0].id
        next_page = await provider.alist_files(limit=1, cursor=first_page.next_cursor, purpose="batch", order="desc")
        assert len(next_page.data) == 1
        assert next_page.data[0].purpose == "batch"
        assert next_page.data[0].id != first_page.data[0].id

        deleted = await provider.adelete_file(uploaded.id)
        file_ids.remove(uploaded.id)
        assert deleted.id == uploaded.id
        assert deleted.deleted is True
        with pytest.raises(ProviderFileNotFoundError) as error:
            await provider.aretrieve_file(uploaded.id)
        assert error.value.status_code == 404
    finally:
        await cleanup_files(provider, file_ids)


@pytest.mark.asyncio
@pytest.mark.parametrize("consumption", ["unread", "early", "full"])
async def test_openai_batch_file_download(consumption: str) -> None:
    provider = create_provider()
    file_id: str | None = None
    try:
        uploaded = await provider.aupload_file(
            BATCH_CONTENT,
            filename=f"any-llm-test-{uuid.uuid4().hex}.jsonl",
            purpose="batch",
            expires_in=3600,
        )
        file_id = uploaded.id
        async with provider.adownload_file(file_id, chunk_size=8) as download:
            assert download.status_code == 200
            assert download.headers.get("content-type")
            if consumption == "early":
                assert await anext(download) == BATCH_CONTENT[:8]
            elif consumption == "full":
                chunks = [chunk async for chunk in download]
                assert all(0 < len(chunk) <= 8 for chunk in chunks)
                assert b"".join(chunks) == BATCH_CONTENT
    finally:
        await cleanup_files(provider, [file_id] if file_id is not None else [])


@pytest.mark.asyncio
async def test_openai_user_data_download_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANY_LLM_UNIFIED_EXCEPTIONS", "1")
    provider = create_provider()
    file_id: str | None = None
    try:
        uploaded = await provider.aupload_file(
            b"any-llm Files test\n",
            filename=f"any-llm-test-{uuid.uuid4().hex}.txt",
            mime_type="text/plain",
            purpose="user_data",
            expires_in=3600,
        )
        file_id = uploaded.id
        with pytest.raises(InvalidRequestError, match="user_data") as error:
            async with provider.adownload_file(file_id):
                pytest.fail("OpenAI must reject the download before entering the context body")
        assert error.value.status_code == 400
    finally:
        await cleanup_files(provider, [file_id] if file_id is not None else [])


def test_openai_files_sync_lifecycle() -> None:
    provider = create_provider()
    file_id: str | None = None
    try:
        uploaded = provider.upload_file(
            BATCH_CONTENT,
            filename=f"any-llm-test-{uuid.uuid4().hex}.jsonl",
            purpose="batch",
            expires_in=3600,
        )
        file_id = uploaded.id
        assert uploaded.size_bytes == len(BATCH_CONTENT)
        assert provider.retrieve_file(file_id).id == file_id
        page = provider.list_files(limit=1, purpose="batch")
        assert len(page.data) == 1
        assert page.data[0].purpose == "batch"
        with provider.download_file(file_id, chunk_size=8) as download:
            assert download.status_code == 200
            assert download.headers.get("content-type")
            chunks = list(download)
            assert all(0 < len(chunk) <= 8 for chunk in chunks)
            assert b"".join(chunks) == BATCH_CONTENT
        with provider.download_file(file_id, chunk_size=8) as download:
            assert next(download) == BATCH_CONTENT[:8]
        deleted = provider.delete_file(file_id)
        file_id = None
        assert deleted.id == uploaded.id
        assert deleted.deleted is True
    finally:
        run_async_in_sync(
            cleanup_files(provider, [file_id] if file_id is not None else [], primary_error=sys.exc_info()[1])
        )
