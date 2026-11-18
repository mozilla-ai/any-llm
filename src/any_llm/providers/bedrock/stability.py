"""Image edit through the Stability AI image services, which Bedrock serves only on InvokeModel.

Request and response bodies: https://docs.aws.amazon.com/bedrock/latest/userguide/stable-image-services.html
"""

from __future__ import annotations

import base64
import re
import time
from pathlib import Path
from typing import Any, Literal

from any_llm.exceptions import ContentFilterError, InvalidRequestError, ProviderError, UnsupportedParameterError
from any_llm.types.image import (
    Image,
    ImageEditParams,
    ImageFile,
    ImagesResponse,
    _validate_image_file,
    read_image_content,
)

_STYLE_TRANSFER = "stable-style-transfer"

# Fast Upscale answers a prompt with "Available fields for upscale-fast are: image, output_format";
# Remove Background and Erase Object document no prompt either.
_PROMPTLESS_SERVICES = frozenset({"stable-fast-upscale", "stable-image-erase-object", "stable-image-remove-background"})

IMAGE_SERVICES = frozenset(
    {
        _STYLE_TRANSFER,
        *_PROMPTLESS_SERVICES,
        "stable-conservative-upscale",
        "stable-creative-upscale",
        "stable-image-control-sketch",
        "stable-image-control-structure",
        "stable-image-inpaint",
        "stable-image-search-recolor",
        "stable-image-search-replace",
        "stable-image-style-guide",
        "stable-outpaint",
    }
)

# Matches the foundation model ID and its inference profiles, such as us.stability.stable-outpaint-v1:0.
_SERVICE_MODEL = re.compile(r"(?:^|\.)stability\.([a-z0-9-]+?)-v\d+(?::\d+)?$")

# InvokeModel adds this field to the body when a guardrail is applied to the request.
_GUARDRAIL_ACTION = "amazon-bedrock-guardrailAction"

_OUTPUT_FORMATS: dict[str, Literal["png", "jpeg", "webp"]] = {"png": "png", "jpeg": "jpeg", "webp": "webp"}


def _service(model_id: str, provider_name: str) -> str:
    match = _SERVICE_MODEL.search(model_id)
    if match is None or match.group(1) not in IMAGE_SERVICES:
        msg = (
            f"{model_id!r} is not a Stability AI image service. Bedrock edits images only with those services, "
            "such as 'us.stability.stable-image-inpaint-v1:0'."
        )
        raise InvalidRequestError(msg, provider_name=provider_name)
    return match.group(1)


def _encode(image: ImageFile, provider_name: str) -> str:
    content = image[1] if isinstance(image, tuple) else image
    if isinstance(content, str):
        content = Path(content)
    return base64.b64encode(read_image_content(content, provider_name)).decode()


def _check_options(params: ImageEditParams, provider_name: str) -> None:
    """Refuse the OpenAI options that would change the result; ``quality`` and ``user`` mean nothing here."""
    if params.n not in (None, 1):
        parameter_name = "n"
        raise UnsupportedParameterError(
            parameter_name, provider_name, "Stability AI image services return one image per request."
        )
    if params.size not in (None, "auto"):
        parameter_name = "size"
        raise UnsupportedParameterError(
            parameter_name,
            provider_name,
            "Stability AI image services set the output size from the input image and the service's own fields.",
        )
    if params.response_format == "url":
        parameter_name = "response_format"
        raise UnsupportedParameterError(
            parameter_name, provider_name, "Bedrock returns the image bytes, so only 'b64_json' is available."
        )


def stability_edit_body(params: ImageEditParams, provider_name: str, **kwargs: Any) -> dict[str, Any]:
    """Build the InvokeModel body for a Stability AI image service.

    Style Transfer takes two images, the one to restyle and then the style image; every other
    service takes one. An empty prompt is left out, since three services take none. ``mask`` is
    an image like ``image``, and every other keyword argument is a field of the body, such as
    ``output_format``, ``seed`` or ``negative_prompt``.
    """
    service = _service(params.model_id, provider_name)
    _check_options(params, provider_name)

    images = params.image if isinstance(params.image, list) else [params.image]
    body: dict[str, Any] = {}
    if service == _STYLE_TRANSFER:
        if len(images) != 2:
            msg = f"{service} takes two images: the image to restyle, then the style image."
            raise InvalidRequestError(msg, provider_name=provider_name)
        body["init_image"] = _encode(images[0], provider_name)
        body["style_image"] = _encode(images[1], provider_name)
    else:
        if len(images) != 1:
            msg = f"{service} takes one image, got {len(images)}."
            raise InvalidRequestError(msg, provider_name=provider_name)
        body["image"] = _encode(images[0], provider_name)

    if params.prompt:
        if service in _PROMPTLESS_SERVICES:
            parameter_name = "prompt"
            raise UnsupportedParameterError(parameter_name, provider_name, f"{service} takes no prompt; pass ''.")
        body["prompt"] = params.prompt

    mask = kwargs.pop("mask", None)
    if mask is not None:
        try:
            _validate_image_file(mask)
        except ValueError as exc:
            msg = "mask must be bytes, a path, a readable binary file object, or a (filename, content, content_type) tuple"
            raise InvalidRequestError(msg, provider_name=provider_name) from exc
        body["mask"] = _encode(mask, provider_name)
    return {**body, **kwargs}


def convert_stability_response(response: dict[str, Any], output_format: Any, provider_name: str) -> ImagesResponse:
    """Read a Stability AI image service response, raising when a filter or guardrail blocked it or it failed."""
    if response.get(_GUARDRAIL_ACTION) == "INTERVENED":
        msg = "A Bedrock guardrail blocked the image edit."
        raise ContentFilterError(msg, provider_name=provider_name)
    finish_reason = next(iter(response.get("finish_reasons") or []), None)
    if isinstance(finish_reason, str) and finish_reason.startswith("Filter reason"):
        msg = f"Stability AI did not return an image: {finish_reason}"
        raise ContentFilterError(msg, provider_name=provider_name)
    if finish_reason is not None:
        msg = f"Stability AI did not return an image: {finish_reason}"
        raise ProviderError(msg, provider_name=provider_name)
    images = response.get("images") or []
    if not images:
        msg = "Stability AI returned no image."
        raise ProviderError(msg, provider_name=provider_name)
    if not all(isinstance(image, str) and image for image in images):
        msg = "Stability AI returned an empty image."
        raise ProviderError(msg, provider_name=provider_name)
    return ImagesResponse(
        created=int(time.time()),
        data=[Image(b64_json=image) for image in images],
        output_format=_OUTPUT_FORMATS.get(output_format or "png"),
    )
