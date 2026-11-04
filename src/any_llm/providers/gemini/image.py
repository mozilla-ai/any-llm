"""Image generation and edit through ``generate_content`` on the Gemini image models (Nano Banana)."""

from __future__ import annotations

import base64
import re
import time
from math import gcd
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from google.genai import types
from openai.types.images_response import Usage, UsageInputTokensDetails, UsageOutputTokensDetails

from any_llm.exceptions import (
    ContentFilterError,
    InvalidRequestError,
    ProviderError,
    UnsupportedParameterError,
)
from any_llm.types.image import Image, ImageEditParams, ImageGenerationParams, ImagesResponse, sniff_image_format

from .base import GoogleProvider

if TYPE_CHECKING:
    from openai._types import FileTypes

# The image models refuse more than one candidate per request ("Multiple candidates is not
# enabled for this model"), so n images take n requests. OpenAI caps n at 10 as well.
MAX_IMAGES = 10

# The ratios Nano Banana 2.1 accepts, from its own error for an unknown one. Older image models
# accept a subset and reject the rest themselves.
_ASPECT_RATIOS = frozenset(
    {"1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"}
)

# OpenAI's sizes that do not reduce to one of Gemini's ratios, mapped to the closest one.
_OPENAI_SIZES = {"1792x1024": "16:9", "1024x1792": "9:16"}

# Gemini image model ids carry "image" as a dash-separated word (gemini-3.1-flash-image,
# gemini-3-pro-image-preview) or name Nano Banana. Imagen models use a different API.
_IMAGE_MODEL = re.compile(r"(^|-)image(-|$)|nano-banana")

_CONTENT_FILTER_REASONS = frozenset(
    {
        types.FinishReason.SAFETY,
        types.FinishReason.PROHIBITED_CONTENT,
        types.FinishReason.BLOCKLIST,
        types.FinishReason.SPII,
        types.FinishReason.IMAGE_SAFETY,
        types.FinishReason.IMAGE_PROHIBITED_CONTENT,
    }
)

_OUTPUT_FORMATS: dict[str, Literal["png", "webp", "jpeg"]] = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/webp": "webp",
}


def is_image_model(model_id: str) -> bool:
    """Return whether ``model_id`` names a Gemini model that answers with images."""
    return bool(_IMAGE_MODEL.search(model_id.removeprefix("models/").lower()))


def _check_image_model(model_id: str, provider_name: str) -> None:
    # A text model asked for images answers with a misleading 400 ("Image generation is not
    # available in your country"), so refuse it before sending anything.
    if not is_image_model(model_id):
        msg = (
            f"{model_id!r} does not generate images. Use a Gemini image model such as "
            "'gemini-nano-banana-2.1' or 'gemini-3.1-flash-image'."
        )
        raise InvalidRequestError(msg, provider_name=provider_name)


def _check_n(n: int | None, provider_name: str) -> int:
    if n is None:
        return 1
    if not 1 <= n <= MAX_IMAGES:
        msg = f"n must be between 1 and {MAX_IMAGES}, got {n}"
        raise InvalidRequestError(msg, provider_name=provider_name)
    return n


def _check_response_format(response_format: str | None, provider_name: str) -> None:
    if response_format == "url":
        parameter_name = "response_format"
        raise UnsupportedParameterError(
            parameter_name, provider_name, "Gemini returns the image bytes, so only 'b64_json' is available."
        )


def _aspect_ratio(size: str | None, provider_name: str) -> str | None:
    """Map an OpenAI ``WIDTHxHEIGHT`` size, or a Gemini ratio such as ``16:9``, to an aspect ratio."""
    if size is None or size == "auto":
        return None
    if size in _ASPECT_RATIOS:
        return size
    if size in _OPENAI_SIZES:
        return _OPENAI_SIZES[size]
    width, separator, height = size.partition("x")
    # isdecimal, not isdigit: isdigit accepts characters such as '²' that int() rejects.
    if separator and width.isdecimal() and height.isdecimal() and int(width) and int(height):
        divisor = gcd(int(width), int(height))
        ratio = f"{int(width) // divisor}:{int(height) // divisor}"
        if ratio in _ASPECT_RATIOS:
            return ratio
    ratios = ", ".join(sorted(_ASPECT_RATIOS, key=lambda r: [int(side) for side in r.split(":")]))
    msg = (
        f"Gemini sets the shape of an image by aspect ratio, and {size!r} matches none of {ratios}. "
        "Pass a WIDTHxHEIGHT that reduces to one of them, one of the ratios itself, or 'auto'."
    )
    raise InvalidRequestError(msg, provider_name=provider_name)


def build_config(
    size: str | None,
    n: int | None,
    response_format: str | None,
    provider_name: str,
    **kwargs: Any,
) -> tuple[types.GenerateContentConfig, int]:
    """Build the ``generate_content`` config for an image request and the number of requests to send.

    Extra keyword arguments that name an ``ImageConfig`` field (``image_size``,
    ``person_generation``) go to ``image_config``, merged with any ``image_config`` passed; the
    rest go to ``GenerateContentConfig``. ``timeout`` (seconds) goes to ``http_options``.
    """
    count = _check_n(n, provider_name)
    _check_response_format(response_format, provider_name)

    if (timeout := kwargs.pop("timeout", None)) is not None:
        GoogleProvider._merge_timeout_into_http_options(timeout, kwargs)
    image_config = kwargs.pop("image_config", None)
    image_fields = (
        {} if image_config is None else types.ImageConfig.model_validate(image_config).model_dump(exclude_none=True)
    )
    image_fields.update({name: kwargs.pop(name) for name in list(kwargs) if name in types.ImageConfig.model_fields})
    if (aspect_ratio := _aspect_ratio(size, provider_name)) is not None:
        if image_fields.get("aspect_ratio", aspect_ratio) != aspect_ratio:
            msg = f"size {size!r} asks for {aspect_ratio}, but aspect_ratio asks for {image_fields['aspect_ratio']}"
            raise InvalidRequestError(msg, provider_name=provider_name)
        image_fields["aspect_ratio"] = aspect_ratio
    unknown = kwargs.keys() - types.GenerateContentConfig.model_fields.keys()
    if unknown:
        parameter_name = min(unknown)
        raise UnsupportedParameterError(parameter_name, provider_name)

    config_fields: dict[str, Any] = {"response_modalities": [types.Modality.IMAGE]}
    if image_fields:
        config_fields["image_config"] = types.ImageConfig(**image_fields)
    config_fields.update(kwargs)
    return types.GenerateContentConfig(**config_fields), count


def generation_request(params: ImageGenerationParams, provider_name: str, **kwargs: Any) -> tuple[dict[str, Any], int]:
    """Translate ImageGenerationParams into ``generate_content`` arguments and a request count.

    ``quality`` and ``style`` have no Gemini counterpart and are dropped, so a request written for
    OpenAI still runs; ``user`` is dropped because Gemini takes no end-user id.
    """
    _check_image_model(params.model_id, provider_name)
    config, count = build_config(params.size, params.n, params.response_format, provider_name, **kwargs)
    return {"model": params.model_id, "contents": params.prompt, "config": config}, count


def _read_bytes(content: Any, provider_name: str) -> bytes:
    if isinstance(content, bytes):
        return content
    if isinstance(content, PathLike):
        path = Path(content)
        try:
            return path.read_bytes()
        except (OSError, ValueError) as exc:
            msg = f"Cannot read image path {str(path)!r}: {getattr(exc, 'strerror', None) or exc}"
            raise InvalidRequestError(msg, original_exception=exc, provider_name=provider_name) from exc
    try:
        data = content.read()
    except (OSError, ValueError) as exc:
        # ValueError is what reading a closed file raises.
        msg = f"Cannot read image file object: {exc}"
        raise InvalidRequestError(msg, original_exception=exc, provider_name=provider_name) from exc
    if not isinstance(data, bytes):
        msg = "Image file objects must be opened in binary mode"
        raise InvalidRequestError(msg, provider_name=provider_name)
    return data


def _image_part(image: FileTypes, provider_name: str) -> types.Part:
    content_type: str | None = None
    if isinstance(image, tuple):
        content = image[1]
        content_type = image[2] if len(image) > 2 else None
    else:
        content = image
    data = _read_bytes(content, provider_name)
    if content_type is None and (sniffed := sniff_image_format(data)) is not None:
        content_type = sniffed[1]
    if content_type is None:
        msg = "Gemini needs the image type: send PNG, JPEG or WEBP bytes, or a (filename, content, content_type) tuple"
        raise InvalidRequestError(msg, provider_name=provider_name)
    return types.Part.from_bytes(data=data, mime_type=content_type)


def edit_request(params: ImageEditParams, provider_name: str, **kwargs: Any) -> tuple[dict[str, Any], int]:
    """Translate ImageEditParams into ``generate_content`` arguments and a request count.

    The images go in as parts before the prompt, the order Gemini's editing examples use.
    """
    _check_image_model(params.model_id, provider_name)
    config, count = build_config(params.size, params.n, params.response_format, provider_name, **kwargs)
    resolved = params.resolve_images()
    images = resolved if isinstance(resolved, list) else [resolved]
    parts = [_image_part(image, provider_name) for image in images]
    contents = types.Content(role="user", parts=[*parts, types.Part.from_text(text=params.prompt)])
    return {"model": params.model_id, "contents": contents, "config": config}, count


def _images(response: types.GenerateContentResponse, provider_name: str) -> list[types.Blob]:
    """The finished images of a response, or the reason there are none."""
    candidate = response.candidates[0] if response.candidates else None
    parts = candidate.content.parts if candidate and candidate.content and candidate.content.parts else []
    # Thinking models can return draft images marked as thoughts before the final one.
    blobs = [
        part.inline_data
        for part in parts
        if part.inline_data is not None
        and part.inline_data.data
        and not part.thought
        and (part.inline_data.mime_type or "").startswith("image/")
    ]
    if blobs:
        return blobs

    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason is not None:
        reason = feedback.block_reason_message or feedback.block_reason.value
        msg = f"Gemini blocked the prompt: {reason}"
        raise ContentFilterError(msg, provider_name=provider_name)
    if candidate is not None and candidate.finish_reason in _CONTENT_FILTER_REASONS:
        reason = candidate.finish_message or candidate.finish_reason.value
        msg = f"Gemini did not return an image: {reason}"
        raise ContentFilterError(msg, provider_name=provider_name)
    text = " ".join(part.text for part in parts if part.text and not part.thought).strip()
    finish = candidate.finish_reason.value if candidate is not None and candidate.finish_reason else "none"
    msg = f"Gemini returned no image (finish reason {finish})" + (f": {text}" if text else "")
    raise ProviderError(msg, provider_name=provider_name)


def _modality_tokens(details: list[types.ModalityTokenCount] | None, modality: types.MediaModality) -> int:
    return sum(detail.token_count or 0 for detail in details or [] if detail.modality == modality)


def _usage(responses: list[types.GenerateContentResponse]) -> Usage | None:
    """Sum the token counts of the responses in the shape OpenAI reports for gpt-image.

    Thinking tokens count as text output, which is how Gemini bills them.
    """
    metadata = [response.usage_metadata for response in responses if response.usage_metadata is not None]
    if not metadata:
        return None
    input_tokens = sum(usage.prompt_token_count or 0 for usage in metadata)
    input_image = sum(_modality_tokens(usage.prompt_tokens_details, types.MediaModality.IMAGE) for usage in metadata)
    output_tokens = sum((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0) for usage in metadata)
    output_image = sum(
        _modality_tokens(usage.candidates_tokens_details, types.MediaModality.IMAGE) for usage in metadata
    )
    total_tokens = sum(usage.total_token_count or 0 for usage in metadata) or input_tokens + output_tokens
    return Usage(
        input_tokens=input_tokens,
        input_tokens_details=UsageInputTokensDetails(image_tokens=input_image, text_tokens=input_tokens - input_image),
        output_tokens=output_tokens,
        output_tokens_details=UsageOutputTokensDetails(
            image_tokens=output_image, text_tokens=output_tokens - output_image
        ),
        total_tokens=total_tokens,
    )


def convert_responses_to_images(responses: list[types.GenerateContentResponse], provider_name: str) -> ImagesResponse:
    """Read ``generate_content`` responses as one ImagesResponse with base64 images and summed usage."""
    blobs = [blob for response in responses for blob in _images(response, provider_name)]
    created = next(
        (int(response.create_time.timestamp()) for response in responses if response.create_time is not None),
        int(time.time()),
    )
    return ImagesResponse(
        created=created,
        data=[Image(b64_json=base64.b64encode(blob.data).decode()) for blob in blobs if blob.data],
        output_format=_OUTPUT_FORMATS.get(blobs[0].mime_type or ""),
        usage=_usage(responses),
    )
