"""Tests for image edit on Bedrock through the Stability AI image services."""

import base64
import io
import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from any_llm.exceptions import ContentFilterError, InvalidRequestError, ProviderError, UnsupportedParameterError
from any_llm.providers.bedrock import BedrockProvider
from any_llm.types.image import read_image_content

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff" + b"\x00" * 16
OUTPUT = b"edited image"

INPAINT = "us.stability.stable-image-inpaint-v1:0"
STYLE_TRANSFER = "us.stability.stable-style-transfer-v1:0"
REMOVE_BACKGROUND = "us.stability.stable-image-remove-background-v1:0"


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _provider(
    finish_reason: str | None = None, images: list[str] | None = None, extra: dict[str, Any] | None = None
) -> tuple[BedrockProvider, Mock]:
    client = Mock()
    body = {
        "seeds": [42],
        "finish_reasons": [finish_reason],
        "images": [_b64(OUTPUT)] if images is None else images,
        **(extra or {}),
    }
    client.invoke_model.side_effect = lambda **_: {"body": io.BytesIO(json.dumps(body).encode())}
    return BedrockProvider(client=client), client


def _sent_body(client: Mock) -> dict[str, Any]:
    return json.loads(client.invoke_model.call_args.kwargs["body"])  # type: ignore[no-any-return]


def test_bedrock_supports_image_edit() -> None:
    assert BedrockProvider.SUPPORTS_IMAGE_EDIT is True


@pytest.mark.asyncio
async def test_image_edit_sends_the_image_and_prompt_to_invoke_model() -> None:
    provider, client = _provider()

    response = await provider.aimage_edit(INPAINT, PNG, "a red hat", output_format="jpeg", seed=7)

    request = client.invoke_model.call_args.kwargs
    assert request["modelId"] == INPAINT
    assert request["contentType"] == "application/json"
    assert request["accept"] == "application/json"
    assert _sent_body(client) == {"image": _b64(PNG), "prompt": "a red hat", "output_format": "jpeg", "seed": 7}
    assert response.data is not None
    assert [image.b64_json for image in response.data] == [_b64(OUTPUT)]
    assert response.output_format == "jpeg"
    assert response.created > 0


@pytest.mark.asyncio
async def test_image_edit_reports_png_when_no_output_format_is_asked() -> None:
    provider, client = _provider()

    response = await provider.aimage_edit(INPAINT, PNG, "a red hat")

    assert "output_format" not in _sent_body(client)
    assert response.output_format == "png"


@pytest.mark.asyncio
async def test_image_edit_leaves_an_unknown_output_format_unreported() -> None:
    provider, client = _provider()

    response = await provider.aimage_edit(INPAINT, PNG, "a red hat", output_format="tiff")

    assert _sent_body(client)["output_format"] == "tiff"
    assert response.output_format is None


@pytest.mark.asyncio
async def test_image_edit_encodes_the_mask() -> None:
    provider, client = _provider()

    await provider.aimage_edit(INPAINT, PNG, "a red hat", mask=JPEG)

    assert _sent_body(client)["mask"] == _b64(JPEG)


@pytest.mark.parametrize("mask", [123, ("mask.jpg", "not file content"), [JPEG]])
@pytest.mark.asyncio
async def test_image_edit_rejects_a_mask_that_is_not_an_image(mask: Any) -> None:
    provider, client = _provider()

    with pytest.raises(InvalidRequestError, match="mask must be"):
        await provider.aimage_edit(INPAINT, PNG, "a red hat", mask=mask)

    client.invoke_model.assert_not_called()


@pytest.mark.asyncio
async def test_image_edit_reads_the_images_off_the_event_loop(tmp_path: Path) -> None:
    path = tmp_path / "photo.png"
    path.write_bytes(PNG)
    provider, client = _provider()
    on_main_thread: list[bool] = []

    def record(content: Any, provider_name: str) -> bytes:
        on_main_thread.append(threading.current_thread() is threading.main_thread())
        return read_image_content(content, provider_name)

    with patch("any_llm.providers.bedrock.stability.read_image_content", side_effect=record):
        await provider.aimage_edit(INPAINT, str(path), "a red hat", mask=JPEG)

    assert on_main_thread == [False, False]
    assert _sent_body(client)["image"] == _b64(PNG)


@pytest.mark.asyncio
async def test_image_edit_reads_paths_tuples_and_file_objects(tmp_path: Path) -> None:
    path = tmp_path / "photo.png"
    path.write_bytes(PNG)
    provider, client = _provider()

    await provider.aimage_edit(INPAINT, str(path), "a red hat", mask=("mask.jpg", io.BytesIO(JPEG), "image/jpeg"))

    assert _sent_body(client)["image"] == _b64(PNG)
    assert _sent_body(client)["mask"] == _b64(JPEG)

    await provider.aimage_edit(INPAINT, io.BytesIO(JPEG), "a red hat")

    assert _sent_body(client)["image"] == _b64(JPEG)


@pytest.mark.asyncio
async def test_image_edit_sends_style_transfer_images_as_init_and_style() -> None:
    provider, client = _provider()

    await provider.aimage_edit(STYLE_TRANSFER, [PNG, JPEG], "")

    assert _sent_body(client) == {"init_image": _b64(PNG), "style_image": _b64(JPEG)}


@pytest.mark.asyncio
async def test_image_edit_rejects_style_transfer_without_two_images() -> None:
    provider, client = _provider()

    with pytest.raises(InvalidRequestError, match="two images"):
        await provider.aimage_edit(STYLE_TRANSFER, PNG, "")

    client.invoke_model.assert_not_called()


@pytest.mark.asyncio
async def test_image_edit_rejects_more_than_one_image_for_other_services() -> None:
    provider, client = _provider()

    with pytest.raises(InvalidRequestError, match="one image"):
        await provider.aimage_edit(INPAINT, [PNG, JPEG], "a red hat")

    client.invoke_model.assert_not_called()


@pytest.mark.asyncio
async def test_image_edit_omits_an_empty_prompt() -> None:
    provider, client = _provider()

    await provider.aimage_edit(REMOVE_BACKGROUND, PNG, "")

    assert _sent_body(client) == {"image": _b64(PNG)}


@pytest.mark.parametrize(
    "model",
    [
        REMOVE_BACKGROUND,
        "us.stability.stable-fast-upscale-v1:0",
        "us.stability.stable-image-erase-object-v1:0",
    ],
)
@pytest.mark.asyncio
async def test_image_edit_rejects_a_prompt_for_services_that_take_none(model: str) -> None:
    provider, client = _provider()

    with pytest.raises(UnsupportedParameterError, match="prompt"):
        await provider.aimage_edit(model, PNG, "remove it")

    client.invoke_model.assert_not_called()


@pytest.mark.parametrize(
    "model",
    [
        "stability.stable-image-control-sketch-v1:0",
        "us.stability.stable-creative-upscale-v1:0",
        "us.stability.stable-conservative-upscale-v1:0",
        "us.stability.stable-outpaint-v1:0",
        "us.stability.stable-image-search-recolor-v1:0",
        "us.stability.stable-image-search-replace-v1:0",
        "us.stability.stable-image-control-structure-v1:0",
        "us.stability.stable-image-style-guide-v1:0",
    ],
)
@pytest.mark.asyncio
async def test_image_edit_accepts_every_prompted_service(model: str) -> None:
    provider, client = _provider()

    await provider.aimage_edit(model, PNG, "a castle")

    assert _sent_body(client) == {"image": _b64(PNG), "prompt": "a castle"}


@pytest.mark.parametrize(
    "model", ["us.anthropic.claude-haiku-4-5-20251001-v1:0", "stability.stable-image-ultra-v1:1", "stability.sd3-5"]
)
@pytest.mark.asyncio
async def test_image_edit_rejects_a_model_that_is_not_an_image_service(model: str) -> None:
    provider, client = _provider()

    with pytest.raises(InvalidRequestError, match="Stability AI image service"):
        await provider.aimage_edit(model, PNG, "a red hat")

    client.invoke_model.assert_not_called()


@pytest.mark.parametrize(
    ("option", "value"),
    [("n", 2), ("size", "1024x1024"), ("response_format", "url")],
)
@pytest.mark.asyncio
async def test_image_edit_rejects_options_the_services_do_not_take(option: str, value: Any) -> None:
    provider, client = _provider()

    with pytest.raises(UnsupportedParameterError, match=option):
        await provider.aimage_edit(INPAINT, PNG, "a red hat", **{option: value})

    client.invoke_model.assert_not_called()


@pytest.mark.asyncio
async def test_image_edit_accepts_the_options_that_mean_one_base64_image() -> None:
    provider, client = _provider()

    await provider.aimage_edit(
        INPAINT, PNG, "a red hat", n=1, size="auto", response_format="b64_json", quality="high", user="u-1"
    )

    assert _sent_body(client) == {"image": _b64(PNG), "prompt": "a red hat"}


@pytest.mark.asyncio
async def test_image_edit_raises_content_filter_error_for_a_filtered_request() -> None:
    provider, client = _provider(finish_reason="Filter reason: output image", images=[])

    with pytest.raises(ContentFilterError, match="Filter reason: output image"):
        await provider.aimage_edit(INPAINT, PNG, "a red hat")

    client.invoke_model.assert_called_once()
    assert client.invoke_model.call_args.kwargs["modelId"] == INPAINT


@pytest.mark.asyncio
async def test_image_edit_raises_provider_error_for_an_inference_error() -> None:
    provider, client = _provider(finish_reason="Inference error", images=[])

    with pytest.raises(ProviderError, match="Inference error"):
        await provider.aimage_edit(INPAINT, PNG, "a red hat")

    client.invoke_model.assert_called_once()
    assert client.invoke_model.call_args.kwargs["modelId"] == INPAINT


@pytest.mark.asyncio
async def test_image_edit_raises_provider_error_when_no_image_comes_back() -> None:
    provider, client = _provider(images=[])

    with pytest.raises(ProviderError, match="no image"):
        await provider.aimage_edit(INPAINT, PNG, "a red hat")

    client.invoke_model.assert_called_once()
    assert client.invoke_model.call_args.kwargs["modelId"] == INPAINT


@pytest.mark.parametrize("image", ["", None])
@pytest.mark.asyncio
async def test_image_edit_raises_provider_error_for_an_empty_image_entry(image: Any) -> None:
    provider, _ = _provider(images=[image])

    with pytest.raises(ProviderError, match="empty image"):
        await provider.aimage_edit(INPAINT, PNG, "a red hat")


@pytest.mark.asyncio
async def test_image_edit_sends_invoke_model_options_beside_the_body() -> None:
    provider, client = _provider()

    await provider.aimage_edit(
        INPAINT,
        PNG,
        "a red hat",
        guardrailIdentifier="gr-1",
        guardrailVersion="DRAFT",
        trace="ENABLED",
        performanceConfigLatency="standard",
        serviceTier="default",
        requestMetadata={"team": "photos"},
        seed=7,
    )

    request = client.invoke_model.call_args.kwargs
    assert request["guardrailIdentifier"] == "gr-1"
    assert request["guardrailVersion"] == "DRAFT"
    assert request["trace"] == "ENABLED"
    assert request["performanceConfigLatency"] == "standard"
    assert request["serviceTier"] == "default"
    assert request["requestMetadata"] == {"team": "photos"}
    assert request["modelId"] == INPAINT
    assert _sent_body(client) == {"image": _b64(PNG), "prompt": "a red hat", "seed": 7}


@pytest.mark.asyncio
async def test_image_edit_raises_content_filter_error_when_a_guardrail_intervenes() -> None:
    provider, client = _provider(images=[], extra={"amazon-bedrock-guardrailAction": "INTERVENED"})

    with pytest.raises(ContentFilterError, match="guardrail"):
        await provider.aimage_edit(INPAINT, PNG, "a red hat", guardrailIdentifier="gr-1", guardrailVersion="1")

    assert client.invoke_model.call_args.kwargs["guardrailIdentifier"] == "gr-1"


@pytest.mark.asyncio
async def test_image_edit_builds_the_timeout_client_off_the_event_loop() -> None:
    provider, _ = _provider()
    on_main_thread: list[bool] = []
    build_client = provider._client_for_timeout

    def record(timeout: float | None) -> Any:
        on_main_thread.append(threading.current_thread() is threading.main_thread())
        return build_client(timeout)

    with patch.object(provider, "_client_for_timeout", side_effect=record):
        await provider.aimage_edit(INPAINT, PNG, "a red hat")

    assert on_main_thread == [False]


@pytest.mark.asyncio
async def test_image_edit_rejects_a_timeout_on_a_custom_client() -> None:
    provider, client = _provider()

    with pytest.raises(UnsupportedParameterError, match="timeout"):
        await provider.aimage_edit(INPAINT, PNG, "a red hat", timeout=5)

    client.invoke_model.assert_not_called()


def test_image_edit_runs_synchronously() -> None:
    provider, client = _provider()

    response = provider._image_edit(INPAINT, PNG, "a red hat")

    assert response.data is not None
    assert response.data[0].b64_json == _b64(OUTPUT)
    client.invoke_model.assert_called_once()
    assert client.invoke_model.call_args.kwargs["modelId"] == INPAINT
    assert _sent_body(client) == {"image": _b64(PNG), "prompt": "a red hat"}
