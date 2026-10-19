import asyncio
import email
import email.message
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from any_llm import AnyLLM
from any_llm.api import aimage_edit, image_edit
from any_llm.constants import LLMProvider
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.image import ImageEditParams, ImagesResponse


def _form_parts(request: httpx.Request) -> list[tuple[str | None, str | None, str, bytes]]:
    """Return each part of a multipart request as ``(name, filename, content_type, payload)``."""
    raw = b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n" + request.content
    parts = []
    for part in email.message_from_bytes(raw).get_payload():
        assert isinstance(part, email.message.Message)
        payload = part.get_payload(decode=True)
        assert isinstance(payload, bytes)
        name = part.get_param("name", header="content-disposition")
        assert name is None or isinstance(name, str)
        parts.append((name, part.get_filename(), part.get_content_type(), payload))
    return parts


def test_image_edit_uploads_each_image_with_the_prompt_and_extra_options() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request.read()
        requests.append(request)
        return httpx.Response(200, json={"created": 1, "data": [{"b64_json": "aW1hZ2U="}]})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = image_edit(
            "openai:gpt-image-1",
            [("cat.png", b"cat", "image/png"), ("hat.webp", b"hat", "image/webp")],
            "Put the hat on the cat",
            size="1024x1024",
            output_format="jpeg",
            api_key="key",
            client_args={"http_client": http_client},
        )
    finally:
        asyncio.run(http_client.aclose())

    assert result.data is not None
    assert result.data[0].b64_json == "aW1hZ2U="
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/images/edits"
    assert sorted(_form_parts(requests[0])) == [
        ("image[]", "cat.png", "image/png", b"cat"),
        ("image[]", "hat.webp", "image/webp", b"hat"),
        ("model", None, "text/plain", b"gpt-image-1"),
        ("output_format", None, "text/plain", b"jpeg"),
        ("prompt", None, "text/plain", b"Put the hat on the cat"),
        ("size", None, "text/plain", b"1024x1024"),
    ]


@pytest.mark.asyncio
async def test_aimage_edit_passes_params_and_extra_options_to_the_provider() -> None:
    image = ("cat.png", b"cat", "image/png")
    mock_provider = Mock()
    mock_provider._aimage_edit = AsyncMock(return_value=ImagesResponse(created=1, data=[]))

    with patch("any_llm.any_llm.AnyLLM.create") as mock_create:
        mock_create.return_value = mock_provider
        await aimage_edit("gpt-image-1", image, "Add a hat", provider="openai", quality="low", output_format="jpeg")

    assert mock_create.call_args[0][0] == LLMProvider.OPENAI
    params, kwargs = mock_provider._aimage_edit.call_args
    assert params == (ImageEditParams(model_id="gpt-image-1", image=image, prompt="Add a hat", quality="low"),)
    assert kwargs == {"output_format": "jpeg"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("supported", "message"),
    [(False, "doesn't support image edit"), (True, "Subclasses must implement _aimage_edit")],
)
async def test_aimage_edit_raises_when_the_provider_has_no_implementation(supported: bool, message: str) -> None:
    base = Mock(spec=AnyLLM)
    base.SUPPORTS_IMAGE_EDIT = supported
    with pytest.raises(NotImplementedError, match=message):
        await AnyLLM._aimage_edit(base, ImageEditParams(model_id="some-model", image=b"cat", prompt="Add a hat"))


@pytest.mark.asyncio
async def test_aimage_edit_raises_on_an_openai_compatible_provider_without_it() -> None:
    provider = Mock(spec=BaseOpenAIProvider)
    provider.SUPPORTS_IMAGE_EDIT = False
    with pytest.raises(NotImplementedError, match="doesn't support image edit"):
        await BaseOpenAIProvider._aimage_edit(
            provider, ImageEditParams(model_id="some-model", image=b"cat", prompt="Add a hat")
        )


def test_only_openai_supports_image_edit() -> None:
    for provider_enum in LLMProvider:
        try:
            cls = AnyLLM.get_provider_class(provider_enum)
        except ImportError:
            continue
        expected = provider_enum is LLMProvider.OPENAI
        assert cls.get_provider_metadata().image_edit is expected, f"{provider_enum.value} image_edit != {expected}"
