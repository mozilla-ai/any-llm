import os
import uuid

import pytest

from any_llm.exceptions import MissingApiKeyError
from any_llm.providers.anthropic.anthropic import AnthropicProvider
from any_llm.types.messages import MessageResponse
from tests.constants import EXPECTED_PROVIDERS


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
            expires_in_seconds=3600,
        )
        file_id = uploaded.id
        assert uploaded.size_bytes == 21
        assert uploaded.downloadable is False
        retrieved = await provider.aretrieve_file(file_id)
        assert retrieved.id == file_id
        page = await provider.alist_files(ids=[file_id])
        assert [file.id for file in page.data] == [file_id]
        assert page.next_page is None
        legacy = await provider.alist_files(limit=1, betas=["files-api-2025-04-14"])
        assert len(legacy.data) <= 1
        deleted = await provider.adelete_file(file_id)
        assert deleted.id == file_id
        absent = await provider.alist_files(ids=[file_id])
        assert absent.data == []
        file_id = None
    finally:
        if file_id is not None:
            await provider.adelete_file(file_id)
        await provider.client.close()


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
            async with provider.adownload_file(file_id, chunk_size=1024) as chunks:
                contents.append(b"".join([chunk async for chunk in chunks]))
        assert b"any-llm Files test\n" in contents
    finally:
        try:
            for file_id in output_ids:
                await provider.adelete_file(file_id)
        finally:
            await provider.client.close()
