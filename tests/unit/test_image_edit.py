import asyncio
import email
import email.message
import io
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from pydantic import ValidationError

from any_llm import AnyLLM
from any_llm.api import aimage_edit, image_edit
from any_llm.constants import LLMProvider
from any_llm.exceptions import InvalidRequestError
from any_llm.providers.azureopenai.azureopenai import AzureopenaiProvider
from any_llm.providers.gmi.gmi import GmiProvider
from any_llm.providers.openai.openai import OpenaiProvider
from any_llm.types.image import ImageEditParams, ImageInput, ImagesResponse

PNG = b"\x89PNG\r\n\x1a\n" + b"png"
JPEG = b"\xff\xd8\xff\xe0" + b"jpeg"
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 "


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


def _run_image_edit(image: ImageInput) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request.read()
        requests.append(request)
        return httpx.Response(200, json={"created": 1, "data": []})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        image_edit(
            "openai:gpt-image-1",
            image,
            "Add a hat",
            api_key="key",
            client_args={"http_client": http_client, "unified_exceptions": True},
        )
    finally:
        asyncio.run(http_client.aclose())
    return requests


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
    provider = OpenaiProvider(api_key="test_key")

    with (
        patch("any_llm.any_llm.AnyLLM.create", return_value=provider) as mock_create,
        patch.object(provider, "_aimage_edit", AsyncMock(return_value=ImagesResponse(created=1, data=[]))) as mock_edit,
    ):
        await aimage_edit("gpt-image-1", image, "Add a hat", provider="openai", quality="low", output_format="jpeg")

    assert mock_create.call_args[0][0] == LLMProvider.OPENAI
    params, kwargs = mock_edit.call_args
    assert params == (ImageEditParams(model_id="gpt-image-1", image=image, prompt="Add a hat", quality="low"),)
    assert kwargs == {"output_format": "jpeg"}


def test_image_edit_passes_params_and_extra_options_to_the_provider() -> None:
    image = ("cat.png", b"cat", "image/png")
    provider = OpenaiProvider(api_key="test_key")

    with (
        patch("any_llm.any_llm.AnyLLM.create", return_value=provider) as mock_create,
        patch.object(provider, "_aimage_edit", AsyncMock(return_value=ImagesResponse(created=1, data=[]))) as mock_edit,
    ):
        image_edit("gpt-image-1", image, "Add a hat", provider="openai", quality="low", output_format="jpeg")

    assert mock_create.call_args[0][0] == LLMProvider.OPENAI
    params, kwargs = mock_edit.call_args
    assert params == (ImageEditParams(model_id="gpt-image-1", image=image, prompt="Add a hat", quality="low"),)
    assert kwargs == {"output_format": "jpeg"}


@pytest.mark.asyncio
async def test_aimage_edit_raises_on_a_provider_without_it_before_sending() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"created": 1, "data": []})

    provider = GmiProvider(api_key="key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    try:
        with pytest.raises(NotImplementedError, match="doesn't support image edit"):
            await provider.aimage_edit("some-model", PNG, "Add a hat")
    finally:
        await provider.client.close()

    assert requests == []


@pytest.mark.asyncio
async def test_aimage_edit_base_raises_when_a_supporting_provider_has_no_implementation() -> None:
    base = Mock(spec=AnyLLM)
    with pytest.raises(NotImplementedError, match="Subclasses must implement _aimage_edit"):
        await AnyLLM._aimage_edit(base, ImageEditParams(model_id="some-model", image=PNG, prompt="Add a hat"))


@pytest.mark.parametrize(
    ("content", "filename", "content_type"),
    [(PNG, "image.png", "image/png"), (JPEG, "image.jpg", "image/jpeg"), (WEBP, "image.webp", "image/webp")],
)
def test_image_edit_uploads_bytes_under_their_image_type(content: bytes, filename: str, content_type: str) -> None:
    requests = _run_image_edit(content)

    assert ("image", filename, content_type, content) in _form_parts(requests[0])


@pytest.mark.parametrize("as_str", [False, True])
def test_image_edit_reads_a_path_and_uploads_it_under_its_name(tmp_path: Path, as_str: bool) -> None:
    path = tmp_path / "cat.png"
    path.write_bytes(PNG)

    requests = _run_image_edit(str(path) if as_str else path)

    assert ("image", "cat.png", "image/png", PNG) in _form_parts(requests[0])


def test_image_edit_reports_a_missing_path_as_an_invalid_request(tmp_path: Path) -> None:
    with pytest.raises(InvalidRequestError, match="Cannot read image path") as exc_info:
        _run_image_edit(tmp_path / "missing.png")

    assert isinstance(exc_info.value.original_exception, FileNotFoundError)


def test_resolve_images_leaves_unrecognized_bytes_and_file_objects_as_given() -> None:
    stream = io.BytesIO(PNG)
    params = ImageEditParams(model_id="gpt-image-1", image=[b"not an image", stream], prompt="Add a hat")

    assert params.resolve_images() == [b"not an image", stream]


def test_resolve_images_names_a_path_the_bytes_do_not_identify(tmp_path: Path) -> None:
    path = tmp_path / "cat.png"
    path.write_bytes(b"not an image")
    params = ImageEditParams(model_id="gpt-image-1", image=path, prompt="Add a hat")

    assert params.resolve_images() == ("cat.png", b"not an image", None)


@pytest.mark.parametrize(
    "image",
    [123, [], ("cat.png", 123), [PNG, None], (b"cat", PNG), ("cat.png", "images/cat.png"), ("a", ("b", PNG))],
)
def test_image_edit_params_rejects_values_that_are_not_images(image: object) -> None:
    with pytest.raises(ValidationError, match="image"):
        ImageEditParams(model_id="gpt-image-1", image=image, prompt="Add a hat")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_aimage_edit_converts_provider_errors_like_the_sync_function() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        error = {"message": "Invalid file type", "type": "invalid_request_error", "code": "unsupported_file_mimetype"}
        return httpx.Response(400, json={"error": error})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(InvalidRequestError) as exc_info:
            await aimage_edit(
                "openai:gpt-image-1",
                b"not an image",
                "Add a hat",
                api_key="key",
                client_args={"http_client": http_client, "unified_exceptions": True},
            )
    finally:
        await http_client.aclose()

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_azure_image_edit_uses_the_preview_api_version() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"created": 1, "data": []})

    provider = AzureopenaiProvider(
        api_key="key",
        api_base="https://resource.azure.com",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    try:
        await provider._aimage_edit(ImageEditParams(model_id="image-deployment", image=PNG, prompt="Add a hat"))
    finally:
        await provider.client.close()

    assert requests[0].url.path == "/openai/v1/images/edits"
    assert requests[0].url.params["api-version"] == "preview"


def test_only_openai_supports_image_edit() -> None:
    for provider_enum in LLMProvider:
        try:
            cls = AnyLLM.get_provider_class(provider_enum)
        except ImportError:
            continue
        expected = provider_enum is LLMProvider.OPENAI
        assert cls.get_provider_metadata().image_edit is expected, f"{provider_enum.value} image_edit != {expected}"
