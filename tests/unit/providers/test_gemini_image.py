import base64
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from google.genai import types

from any_llm.exceptions import ContentFilterError, InvalidRequestError, ProviderError, UnsupportedParameterError
from any_llm.providers.gemini import GeminiProvider
from any_llm.providers.gemini.image import (
    build_config,
    convert_responses_to_images,
    edit_request,
    generation_request,
    is_image_model,
)
from any_llm.types.image import ImageEditParams, ImageGenerationParams, ImagesResponse

MODEL = "gemini-nano-banana-2.1"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 8


def _response(
    parts: list[types.Part] | None = None,
    *,
    finish_reason: types.FinishReason | None = types.FinishReason.STOP,
    usage: types.GenerateContentResponseUsageMetadata | None = None,
    **overrides: Any,
) -> types.GenerateContentResponse:
    if parts is None:
        parts = [types.Part.from_bytes(data=JPEG, mime_type="image/jpeg")]
    candidate = types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish_reason)
    return types.GenerateContentResponse(candidates=[candidate], usage_metadata=usage, **overrides)


def _usage(**overrides: Any) -> types.GenerateContentResponseUsageMetadata:
    fields: dict[str, Any] = {
        "prompt_token_count": 1132,
        "prompt_tokens_details": [
            types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=12),
            types.ModalityTokenCount(modality=types.MediaModality.IMAGE, token_count=1120),
        ],
        "candidates_token_count": 1335,
        "candidates_tokens_details": [types.ModalityTokenCount(modality=types.MediaModality.IMAGE, token_count=1120)],
        "thoughts_token_count": 990,
        "total_token_count": 3457,
    }
    fields.update(overrides)
    return types.GenerateContentResponseUsageMetadata(**fields)


def _generation(**overrides: Any) -> ImageGenerationParams:
    fields: dict[str, Any] = {"model_id": MODEL, "prompt": "a red apple"}
    fields.update(overrides)
    return ImageGenerationParams(**fields)


def _edit(image: Any = PNG, **overrides: Any) -> ImageEditParams:
    fields: dict[str, Any] = {"model_id": MODEL, "image": image, "prompt": "make it green"}
    fields.update(overrides)
    return ImageEditParams(**fields)


def make_provider() -> tuple[GeminiProvider, Mock]:
    with patch("any_llm.providers.gemini.gemini.genai.Client") as client_class:
        client = client_class.return_value
        provider = GeminiProvider(api_key="test-key")
    return provider, client


@pytest.mark.parametrize(
    "model_id",
    [
        "gemini-nano-banana-2.1",
        "nano-banana-pro-preview",
        "gemini-3.1-flash-image",
        "gemini-3-pro-image-preview",
        "gemini-3.1-flash-lite-image",
        "models/gemini-2.5-flash-image",
    ],
)
def test_is_image_model_accepts_the_gemini_image_models(model_id: str) -> None:
    assert is_image_model(model_id)


@pytest.mark.parametrize("model_id", ["gemini-3.8-flash", "gemini-omni-1.1-flash", "imagen-4.0-generate-001"])
def test_is_image_model_rejects_text_video_and_imagen_models(model_id: str) -> None:
    assert not is_image_model(model_id)


def test_generation_request_asks_for_one_image_from_the_prompt() -> None:
    request, count = generation_request(_generation(), "gemini")

    assert count == 1
    assert request == {
        "model": MODEL,
        "contents": "a red apple",
        "config": types.GenerateContentConfig(response_modalities=[types.Modality.IMAGE]),
    }


def test_generation_request_drops_quality_style_and_user() -> None:
    request, _ = generation_request(_generation(quality="hd", style="vivid", user="user-1"), "gemini")

    assert request["config"] == types.GenerateContentConfig(response_modalities=[types.Modality.IMAGE])


@pytest.mark.parametrize(
    ("size", "aspect_ratio"),
    [
        ("1024x1024", "1:1"),
        ("1536x1024", "3:2"),
        ("1024x1536", "2:3"),
        ("1792x1024", "16:9"),
        ("1024x1792", "9:16"),
        ("1920x1080", "16:9"),
        ("21:9", "21:9"),
    ],
)
def test_generation_request_maps_size_to_an_aspect_ratio(size: str, aspect_ratio: str) -> None:
    request, _ = generation_request(_generation(size=size), "gemini")

    assert request["config"].image_config == types.ImageConfig(aspect_ratio=aspect_ratio)


def test_generation_request_leaves_auto_size_to_the_model() -> None:
    request, _ = generation_request(_generation(size="auto"), "gemini")

    assert request["config"].image_config is None


@pytest.mark.parametrize("size", ["1000x700", "big", "0x1024", "1024x", "7:5"])
def test_generation_request_rejects_a_size_with_no_matching_ratio(size: str) -> None:
    with pytest.raises(InvalidRequestError, match="aspect ratio"):
        generation_request(_generation(size=size), "gemini")


def test_generation_request_sends_n_requests() -> None:
    _, count = generation_request(_generation(n=3), "gemini")

    assert count == 3


@pytest.mark.parametrize("n", [0, 11])
def test_generation_request_rejects_n_out_of_range(n: int) -> None:
    with pytest.raises(InvalidRequestError, match="between 1 and 10"):
        generation_request(_generation(n=n), "gemini")


def test_generation_request_accepts_b64_json_and_rejects_url() -> None:
    generation_request(_generation(response_format="b64_json"), "gemini")

    with pytest.raises(UnsupportedParameterError, match="only 'b64_json'"):
        generation_request(_generation(response_format="url"), "gemini")


def test_generation_request_refuses_a_text_model_before_sending() -> None:
    with pytest.raises(InvalidRequestError, match="'gemini-3.8-flash' does not generate images"):
        generation_request(_generation(model_id="gemini-3.8-flash"), "gemini")


def test_build_config_routes_extra_options_to_image_config_and_generate_config() -> None:
    config, _ = build_config(
        "1536x1024",
        None,
        None,
        "gemini",
        image_size="2K",
        seed=7,
        response_modalities=[types.Modality.TEXT, types.Modality.IMAGE],
    )

    assert config.image_config == types.ImageConfig(aspect_ratio="3:2", image_size="2K")
    assert config.seed == 7
    assert config.response_modalities == [types.Modality.TEXT, types.Modality.IMAGE]


def test_build_config_rejects_an_unknown_option() -> None:
    with pytest.raises(UnsupportedParameterError, match="'background' is not supported for gemini"):
        build_config(None, None, None, "gemini", background="transparent")


def test_edit_request_sends_the_image_before_the_prompt() -> None:
    request, count = edit_request(_edit(size="1024x1024"), "gemini")

    assert count == 1
    assert request["model"] == MODEL
    assert request["config"].image_config == types.ImageConfig(aspect_ratio="1:1")
    image_part, text_part = request["contents"].parts
    assert image_part.inline_data == types.Blob(data=PNG, mime_type="image/png")
    assert text_part.text == "make it green"


def test_edit_request_sends_every_image_in_order(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    path.write_bytes(JPEG)
    images = [PNG, path, io.BytesIO(JPEG), ("mask.webp", b"webp-bytes", "image/webp"), ("raw", io.BytesIO(PNG))]

    request, _ = edit_request(_edit(images), "gemini")

    blobs = [part.inline_data for part in request["contents"].parts[:-1]]
    assert blobs == [
        types.Blob(data=PNG, mime_type="image/png"),
        types.Blob(data=JPEG, mime_type="image/jpeg"),
        types.Blob(data=JPEG, mime_type="image/jpeg"),
        types.Blob(data=b"webp-bytes", mime_type="image/webp"),
        types.Blob(data=PNG, mime_type="image/png"),
    ]


def test_edit_request_reads_a_path_inside_a_tuple(tmp_path: Path) -> None:
    path = tmp_path / "photo.png"
    path.write_bytes(PNG)

    request, _ = edit_request(_edit(("photo.png", path)), "gemini")

    assert request["contents"].parts[0].inline_data == types.Blob(data=PNG, mime_type="image/png")


def test_edit_request_reports_an_unreadable_path_inside_a_tuple(tmp_path: Path) -> None:
    with pytest.raises(InvalidRequestError, match="Cannot read image path"):
        edit_request(_edit(("photo.png", tmp_path / "missing.png")), "gemini")


def test_edit_request_rejects_a_text_mode_file() -> None:
    with pytest.raises(InvalidRequestError, match="binary mode"):
        edit_request(_edit(io.StringIO("not bytes")), "gemini")


def test_edit_request_rejects_bytes_of_an_unknown_type() -> None:
    with pytest.raises(InvalidRequestError, match="needs the image type"):
        edit_request(_edit(b"GIF89a"), "gemini")


def test_edit_request_refuses_a_text_model() -> None:
    with pytest.raises(InvalidRequestError, match="does not generate images"):
        edit_request(_edit(model_id="gemini-3.8-flash"), "gemini")


def test_convert_responses_to_images_returns_base64_images_and_usage() -> None:
    response = _response(usage=_usage(), create_time=datetime(2026, 11, 4, tzinfo=UTC))

    result = convert_responses_to_images([response], "gemini")

    assert result.created == int(datetime(2026, 11, 4, tzinfo=UTC).timestamp())
    assert result.data is not None
    assert [image.b64_json for image in result.data] == [base64.b64encode(JPEG).decode()]
    assert result.output_format == "jpeg"
    assert result.usage is not None
    assert result.usage.model_dump() == {
        "input_tokens": 1132,
        "input_tokens_details": {"image_tokens": 1120, "text_tokens": 12},
        "output_tokens": 2325,
        "output_tokens_details": {"image_tokens": 1120, "text_tokens": 1205},
        "total_tokens": 3457,
    }


def test_convert_responses_to_images_sums_several_responses() -> None:
    usage = _usage(prompt_tokens_details=None, total_token_count=None)
    responses = [_response(usage=usage), _response([types.Part.from_bytes(data=PNG, mime_type="image/png")])]

    result = convert_responses_to_images(responses, "gemini")

    assert result.data is not None
    assert len(result.data) == 2
    assert result.usage is not None
    assert result.usage.input_tokens_details.image_tokens == 0
    assert result.usage.total_tokens == 1132 + 1335 + 990


def test_convert_responses_to_images_skips_draft_images_and_text() -> None:
    draft = types.Part(inline_data=types.Blob(data=PNG, mime_type="image/png"), thought=True)
    parts = [types.Part.from_text(text="Here it is"), draft, types.Part.from_bytes(data=JPEG, mime_type="image/jpeg")]

    result = convert_responses_to_images([_response(parts)], "gemini")

    assert result.data is not None
    assert [image.b64_json for image in result.data] == [base64.b64encode(JPEG).decode()]
    assert result.usage is None


def test_convert_responses_to_images_reports_a_blocked_prompt() -> None:
    feedback = types.GenerateContentResponsePromptFeedback(block_reason=types.BlockedReason.SAFETY)
    response = types.GenerateContentResponse(prompt_feedback=feedback)

    with pytest.raises(ContentFilterError, match="blocked the prompt: SAFETY"):
        convert_responses_to_images([response], "gemini")


def test_convert_responses_to_images_reports_a_filtered_image() -> None:
    response = _response([], finish_reason=types.FinishReason.IMAGE_SAFETY)

    with pytest.raises(ContentFilterError, match="did not return an image: IMAGE_SAFETY"):
        convert_responses_to_images([response], "gemini")


def test_convert_responses_to_images_reports_text_without_an_image() -> None:
    response = _response([types.Part.from_text(text="I can only describe it.")])

    with pytest.raises(ProviderError, match=r"no image \(finish reason STOP\): I can only describe it\."):
        convert_responses_to_images([response], "gemini")


def test_convert_responses_to_images_reports_an_empty_response() -> None:
    with pytest.raises(ProviderError, match=r"no image \(finish reason none\)$"):
        convert_responses_to_images([types.GenerateContentResponse()], "gemini")


@pytest.mark.asyncio
async def test_aimage_generation_sends_one_request_per_image() -> None:
    provider, client = make_provider()
    client.aio.models.generate_content = AsyncMock(return_value=_response(usage=_usage()))

    result = await provider.aimage_generation(MODEL, "a red apple", n=2, size="1536x1024", image_size="1K")

    assert isinstance(result, ImagesResponse)
    assert result.data is not None
    assert len(result.data) == 2
    assert result.usage is not None
    assert result.usage.output_tokens == 2 * 2325
    assert client.aio.models.generate_content.await_count == 2
    call = client.aio.models.generate_content.await_args
    assert call.kwargs["model"] == MODEL
    assert call.kwargs["contents"] == "a red apple"
    assert call.kwargs["config"].image_config == types.ImageConfig(aspect_ratio="3:2", image_size="1K")


@pytest.mark.asyncio
async def test_aimage_edit_sends_the_reference_image() -> None:
    provider, client = make_provider()
    client.aio.models.generate_content = AsyncMock(return_value=_response())

    result = await provider.aimage_edit(MODEL, PNG, "make it green")

    assert result.data is not None
    assert result.data[0].b64_json == base64.b64encode(JPEG).decode()
    contents = client.aio.models.generate_content.await_args.kwargs["contents"]
    assert contents.parts[0].inline_data == types.Blob(data=PNG, mime_type="image/png")
    assert contents.parts[1].text == "make it green"


@pytest.mark.asyncio
async def test_aimage_generation_refuses_a_text_model_without_calling_gemini() -> None:
    provider, client = make_provider()
    client.aio.models.generate_content = AsyncMock()

    with pytest.raises(InvalidRequestError, match="does not generate images"):
        await provider.aimage_generation("gemini-3.8-flash", "a cat")

    client.aio.models.generate_content.assert_not_awaited()
