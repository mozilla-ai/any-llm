import base64
import struct
import zlib
from typing import Any

import pytest

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.exceptions import MissingApiKeyError
from tests.constants import EXPECTED_PROVIDERS

pytestmark = [pytest.mark.asyncio, pytest.mark.parametrize("provider", [LLMProvider.BEDROCK])]

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png(side: int) -> bytes:
    """A square RGB PNG, red on the left half and white on the right; the services refuse sides under 64 pixels."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    row = b"\x00" + b"\xff\x00\x00" * (side // 2) + b"\xff\xff\xff" * (side - side // 2)
    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return PNG_SIGNATURE + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(row * side)) + chunk(b"IEND", b"")


def _client(provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]) -> AnyLLM:
    try:
        return AnyLLM.create(provider, **provider_client_config.get(provider, {}))
    except MissingApiKeyError:
        if provider in EXPECTED_PROVIDERS:
            raise
        pytest.skip("Bedrock credentials missing")


async def test_stability_image_service_edits_an_image(
    provider: LLMProvider, provider_client_config: dict[LLMProvider, dict[str, Any]]
) -> None:
    response = await _client(provider, provider_client_config).aimage_edit(
        "us.stability.stable-image-remove-background-v1:0", _png(128), "", output_format="png"
    )

    assert response.data
    assert response.data[0].b64_json
    assert base64.b64decode(response.data[0].b64_json).startswith(PNG_SIGNATURE)
    assert response.output_format == "png"
