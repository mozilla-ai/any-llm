import asyncio
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from any_llm.providers.azureopenai.azureopenai import AzureopenaiProvider
from any_llm.providers.openai.openai import OpenaiProvider
from tests.integration import test_openai_files as lifecycle


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_class", [OpenaiProvider, AzureopenaiProvider])
@pytest.mark.parametrize("synchronous", [False, True])
@pytest.mark.parametrize("acknowledgement", ["success", "not_deleted", "wrong_id"])
async def test_lifecycle_retains_file_for_cleanup_until_deletion_is_verified(
    provider_class: type[OpenaiProvider] | type[AzureopenaiProvider],
    synchronous: bool,
    acknowledgement: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files: dict[str, dict[str, Any]] = {}
    delete_calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        file_id = request.url.path.rsplit("/", 1)[-1]
        if request.method == "POST":
            filename = re.search(rb'filename="([^"]+)"', request.content)
            assert filename is not None
            file_id = f"file-{len(files) + 1}"
            files[file_id] = {
                "id": file_id,
                "object": "file",
                "filename": filename[1].decode(),
                "bytes": len(lifecycle.BATCH_CONTENT),
                "purpose": "batch",
                "created_at": 1700000000,
                "expires_at": 1700000000 + lifecycle.BATCH_EXPIRY_SECONDS,
            }
            return httpx.Response(200, json=files[file_id])
        if request.method == "DELETE":
            delete_calls.append(file_id)
            if len(delete_calls) == 1 and acknowledgement != "success":
                return httpx.Response(
                    200,
                    json={
                        "id": "file-wrong" if acknowledgement == "wrong_id" else file_id,
                        "object": "file",
                        "deleted": acknowledgement != "not_deleted",
                    },
                )
            del files[file_id]
            return httpx.Response(200, json={"id": file_id, "object": "file", "deleted": True})
        if file_id == "content":
            return httpx.Response(200, content=lifecycle.BATCH_CONTENT, headers={"content-type": "application/jsonl"})
        if file_id == "files":
            data = list(reversed(list(files.values())))
            if "after" in request.url.params:
                data = data[1:]
            return httpx.Response(200, json={"data": data[:1], "has_more": len(data) > 1})
        if file_id not in files:
            return httpx.Response(404, json={"error": {"message": "File not found"}})
        return httpx.Response(200, json=files[file_id])

    provider = provider_class(
        api_key="test-key",
        api_base="https://files.test/openai/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )

    async def run_lifecycle() -> None:
        if synchronous:
            await asyncio.to_thread(lifecycle.test_openai_files_sync_lifecycle, provider)
        else:
            await lifecycle.test_openai_files_lifecycle(provider, tmp_path, monkeypatch)

    if acknowledgement == "success":
        await run_lifecycle()
    else:
        with pytest.raises(AssertionError):
            await run_lifecycle()
    assert delete_calls.count("file-1") == (1 if acknowledgement == "success" else 2)
    assert files == {}
    assert provider.client.is_closed()
