from typing import Any
from unittest.mock import AsyncMock

import pytest

from any_llm._files import FilesMixin
from any_llm.types.files import FileMetadata, FilePage


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_public_upload_forwards_neutral_contract(synchronous: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FilesMixin()
    provider.PROVIDER_NAME = "test"
    metadata = FileMetadata(id="opaque-id", purpose="batch", status="ready", size_bytes=12)
    upload = AsyncMock(return_value=metadata)
    monkeypatch.setattr(provider, "_aupload_file", upload)
    options: dict[str, Any] = {
        "filename": "input.jsonl",
        "mime_type": "application/jsonl",
        "purpose": "batch",
        "expires_in": 7200,
        "custom_option": "value",
    }
    if synchronous:
        result = provider.upload_file(b"data", allow_running_loop=True, **options)
    else:
        result = await provider.aupload_file(b"data", **options)
    assert result == metadata
    upload.assert_awaited_once_with(b"data", **options)


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_public_pagination_uses_only_common_cursor(synchronous: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = FilesMixin()
    provider.PROVIDER_NAME = "test"
    listing = AsyncMock(
        side_effect=[
            FilePage(data=[FileMetadata(id="first")], next_cursor="opaque-token"),
            FilePage(data=[FileMetadata(id="second")]),
        ]
    )
    monkeypatch.setattr(provider, "_alist_files", listing)
    cursor = None
    ids: list[str] = []
    while True:
        if synchronous:
            page = provider.list_files(limit=1, cursor=cursor, purpose="batch", allow_running_loop=True)
        else:
            page = await provider.alist_files(limit=1, cursor=cursor, purpose="batch")
        ids.extend(item.id for item in page.data)
        cursor = page.next_cursor
        if cursor is None:
            break
    assert ids == ["first", "second"]
    assert listing.await_count == 2
    assert listing.await_args_list[0].kwargs == {"limit": 1, "cursor": None, "purpose": "batch"}
    assert listing.await_args_list[1].kwargs == {"limit": 1, "cursor": "opaque-token", "purpose": "batch"}


def test_metadata_core_is_provider_neutral_and_preserves_extras() -> None:
    metadata = FileMetadata.model_validate(
        {"id": "opaque-id", "purpose": "batch", "status": "processing", "native_field": "preserved"}
    )
    assert metadata.size_bytes is None
    assert metadata.created_at is None
    assert metadata.expires_at is None
    assert metadata.downloadable is None
    assert metadata.model_extra == {"native_field": "preserved"}
    assert set(FileMetadata.model_fields) == {
        "id",
        "filename",
        "size_bytes",
        "mime_type",
        "created_at",
        "expires_at",
        "purpose",
        "status",
        "downloadable",
    }
    assert set(FilePage.model_fields) == {"data", "next_cursor"}
