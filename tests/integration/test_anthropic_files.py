import os
import sys
import uuid
from collections.abc import Iterable

import pytest

from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.anthropic.anthropic import AnthropicProvider
from any_llm.types.messages import MessageResponse
from tests.constants import EXPECTED_PROVIDERS


async def cleanup_files(provider: AnthropicProvider, file_ids: Iterable[str]) -> None:
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
async def test_anthropic_files_lifecycle() -> None:
    try:
        provider = AnthropicProvider(max_retries=0)
    except MissingApiKeyError:
        if "anthropic" in EXPECTED_PROVIDERS:
            raise
        pytest.skip("ANTHROPIC_API_KEY is not configured")
    file_id: str | None = None
    try:
        uploaded = await provider.aupload_file(
            b"name,value\nexample,1\n",
            filename=f"any-llm-test-{uuid.uuid4().hex}.csv",
            mime_type="text/csv",
            expires_in=3600,
        )
        file_id = uploaded.id
        assert uploaded.size_bytes == 21
        assert uploaded.downloadable is False
        retrieved = await provider.aretrieve_file(file_id)
        assert retrieved.id == file_id
        page = await provider.alist_files(ids=[file_id])
        assert [file.id for file in page.data] == [file_id]
        assert page.next_cursor is None
        first_page = await provider.alist_files(limit=1)
        assert len(first_page.data) <= 1
        if first_page.next_cursor is not None:
            next_page = await provider.alist_files(limit=1, cursor=first_page.next_cursor)
            assert len(next_page.data) <= 1
        deleted = await provider.adelete_file(file_id)
        deleted_file_id = file_id
        file_id = None
        assert deleted.id == deleted_file_id
        absent = await provider.alist_files(ids=[deleted_file_id])
        assert absent.data == []
    finally:
        await cleanup_files(provider, [file_id] if file_id is not None else [])


@pytest.mark.asyncio
async def test_anthropic_generated_file_download() -> None:
    try:
        provider = AnthropicProvider(max_retries=0)
    except MissingApiKeyError:
        if "anthropic" in EXPECTED_PROVIDERS:
            raise
        pytest.skip("ANTHROPIC_API_KEY is not configured")
    output_ids: set[str] = set()
    try:
        response = await provider.amessages(
            model=os.environ.get("ANTHROPIC_FILES_TEST_MODEL", "claude-sonnet-4-6"),
            max_tokens=2048,
            tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
            messages=[
                {
                    "role": "user",
                    "content": "Use code execution to create /mnt/data/any_llm_test.txt containing exactly 'any-llm Files test' followed by a newline. Return the downloadable file.",
                }
            ],
            timeout=90,
        )

        def collect(value: object) -> None:
            if isinstance(value, dict):
                if isinstance(value.get("file_id"), str):
                    output_ids.add(value["file_id"])
                for nested in value.values():
                    collect(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect(nested)

        assert isinstance(response, MessageResponse)
        collect(response.model_dump())
        assert output_ids, "Native code execution returned no generated file references"
        contents: list[bytes] = []
        for file_id in output_ids:
            metadata = await provider.aretrieve_file(file_id)
            assert metadata.downloadable is True
            async with provider.adownload_file(file_id, chunk_size=1024) as download:
                assert download.status_code == 200
                assert download.headers.get("content-type")
                contents.append(b"".join([chunk async for chunk in download]))
        assert b"any-llm Files test\n" in contents
    finally:
        await cleanup_files(provider, output_ids)
