"""Video generation through the Gemini Interactions API (Gemini Omni Flash)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from any_llm.exceptions import InvalidRequestError, ProviderError, UnsupportedParameterError
from any_llm.types.video import VideoError, VideoJob, VideoOutput, VideoStatus

from .interactions import _iso_to_epoch

if TYPE_CHECKING:
    from google.genai._gaos.types.interactions import Interaction

    from any_llm.types.video import VideoGenerationParams

# Omni Flash renders landscape or portrait only, and names a resolution by its short side.
# https://ai.google.dev/gemini-api/docs/omni
_RESOLUTIONS: dict[int, str] = {360: "360p", 720: "720p", 1080: "1080p", 2160: "4k"}
_ASPECT_RATIOS: dict[str, float] = {"16:9": 16 / 9, "9:16": 9 / 16}

# Every interaction status that is not still running has stopped producing a video.
# https://ai.google.dev/api/interactions-api-v1#Interaction
_STATUS_MAP: dict[str, VideoStatus] = {
    "queued": "queued",
    "in_progress": "in_progress",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "failed",
    "incomplete": "failed",
    "requires_action": "failed",
    "budget_exceeded": "failed",
}


def _image_part(image: str) -> dict[str, Any]:
    """An Interactions image part from an http(s) URL, a Files API URI, or a data URL."""
    if not image.startswith("data:"):
        return {"type": "image", "uri": image}
    header, separator, data = image.partition(",")
    mime_type, _, encoding = header.removeprefix("data:").partition(";")
    if not separator or encoding != "base64" or not mime_type:
        msg = "Image data URLs must look like data:image/png;base64,<data>"
        raise InvalidRequestError(msg)
    return {"type": "image", "data": data, "mime_type": mime_type}


def _response_format(params: VideoGenerationParams) -> dict[str, Any]:
    response_format: dict[str, Any] = {"type": "video"}
    shape = params.parse_shape()
    if shape is None:
        return response_format
    if shape.ratio is not None:
        aspect_ratio = next((name for name, value in _ASPECT_RATIOS.items() if math.isclose(shape.ratio, value)), None)
        if aspect_ratio is None:
            msg = f"Gemini renders 16:9 or 9:16 video, not {shape.described!r}"
            raise InvalidRequestError(msg)
        response_format["aspect_ratio"] = aspect_ratio
    if shape.short_side is not None:
        resolution = _RESOLUTIONS.get(shape.short_side)
        if resolution is None:
            sizes = ", ".join(str(side) for side in _RESOLUTIONS)
            msg = f"Gemini renders video with a short side of {sizes} pixels, not {shape.described!r}"
            raise InvalidRequestError(msg)
        response_format["resolution"] = resolution
    return response_format


def convert_video_params(
    params: VideoGenerationParams, provider_name: str, *, api_version: str | None
) -> dict[str, Any]:
    """Translate VideoGenerationParams into ``interactions.create`` arguments.

    The job runs in the background so the call returns at once; ``retrieve_video`` then reads
    the interaction. Omni Flash assigns frame roles by position: the first image part is the
    first frame, the second is the last frame, and reference images follow them.
    """
    if params.seconds is not None:
        # The Interactions SDK has a duration field, but the Gemini API documents no accepted values.
        parameter_name = "seconds"
        raise UnsupportedParameterError(parameter_name, provider_name)
    if params.negative_prompt is not None:
        parameter_name = "negative_prompt"
        raise UnsupportedParameterError(parameter_name, provider_name)
    if params.reference_audios is not None:
        # Omni ignores audio in references and documents audio uploads as unsupported.
        parameter_name = "reference_audios"
        raise UnsupportedParameterError(parameter_name, provider_name)
    if params.reference_videos is not None:
        # Omni reads a video part as a clip to edit, extend or reference, by the prompt or its `task`
        # field, so a reference role needs handling of its own before it can be mapped.
        parameter_name = "reference_videos"
        raise UnsupportedParameterError(parameter_name, provider_name)
    if params.generate_audio is not None:
        # Omni always tries to add a sound track and has no field to turn it off; only the prompt steers it.
        parameter_name = "generate_audio"
        raise UnsupportedParameterError(parameter_name, provider_name)

    parts: list[dict[str, Any]] = []
    if params.first_frame is not None:
        parts.append(_image_part(params.first_frame))
    if params.last_frame is not None:
        parts.append(_image_part(params.last_frame))
    parts.extend(_image_part(image) for image in params.reference_images or [])

    create_kwargs: dict[str, Any] = {
        "model": params.model_id,
        "input": params.prompt if not parts else [*parts, {"type": "text", "text": params.prompt}],
        "background": True,
        "response_format": _response_format(params),
    }
    if api_version is not None:
        create_kwargs["api_version"] = api_version
    if params.seed is not None:
        create_kwargs["generation_config"] = {"seed": params.seed}
    return create_kwargs


def convert_interaction_to_video_job(interaction: Interaction, *, fallback_model: str) -> VideoJob:
    """Read an interaction as a VideoJob, with its video as the single output once it is done."""
    if not interaction.id:
        msg = "Gemini returned an interaction without an id"
        raise ProviderError(msg)
    status = _STATUS_MAP.get(str(interaction.status))
    if status is None:
        msg = f"Gemini returned an unknown interaction status {interaction.status!r} for {interaction.id}"
        raise ProviderError(msg)

    outputs: list[VideoOutput] = []
    if interaction.output_video is not None:
        video = interaction.output_video
        outputs.append(VideoOutput(url=video.uri, b64_data=video.data, mime_type=str(video.mime_type or "video/mp4")))

    error: VideoError | None = None
    if interaction.errors:
        first = interaction.errors[0]
        error = VideoError(message=first.message or "Video generation failed.", code=first.code)
    elif status == "failed":
        error = VideoError(message=f"Video generation ended with status {interaction.status}.")

    created = _iso_to_epoch(interaction.created)
    return VideoJob(
        id=interaction.id,
        model=str(interaction.model or fallback_model),
        status=status,
        created_at=int(created) if created else None,
        error=error,
        outputs=outputs,
    )
