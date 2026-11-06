"""Video generation through DeepInfra's native inference API.

DeepInfra gives every video model its own request schema and publishes it at
``GET /models/{model}``, so the shared any-llm fields are mapped by what that schema has:
Seedance spells a frame ``first_frame_image`` and a resolution ``720p``, Wan 3.0 takes every
image as a typed ``media`` item and spells it ``720P``, Wan 2.6 takes a ``size`` of ``1280*720``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn
from uuid import uuid4

from any_llm.exceptions import InvalidRequestError, ProviderError, UnsupportedParameterError
from any_llm.types.video import VideoError, VideoJob, VideoOutput, VideoStatus

if TYPE_CHECKING:
    from any_llm.types.video import VideoGenerationParams

# Seedance's own sizes are not exact ratios (864x480 is 1.8, not 16:9), so a size counts as a
# listed ratio when it is within a few percent of it.
_ASPECT_RATIO_TOLERANCE = 0.03

# The names each kind of input goes by across DeepInfra's video schemas, in preference order.
_FRAME_FIELDS: dict[str, tuple[str, ...]] = {
    "first_frame": ("first_frame_image", "first_frame"),
    "last_frame": ("last_frame_image", "last_frame"),
}
_REFERENCE_AUDIOS_FIELD = "reference_audios"
# Wan 2.6 and 2.7 take a single clip under a field of their own.
_SINGLE_AUDIO_FIELDS = ("audio_url", "driving_audio")
_RATIO_FIELDS = ("aspect_ratio", "ratio")
# Seedance and Veo, Pixverse 6, then Wan 3.0. p-video's ``audio`` is a clip URL, so only a boolean counts.
_GENERATE_AUDIO_FIELDS = ("generate_audio", "generate_audio_switch", "audio")

# DeepInfra's ``inference_status.status`` values. A synchronous call returns ``succeeded``; the
# others appear with a webhook or when the render fails.
_STATUS_MAP: dict[str, VideoStatus] = {
    "unknown": "completed",
    "queued": "queued",
    "running": "in_progress",
    "succeeded": "completed",
    "failed": "failed",
}


def _ratio_value(name: str) -> float | None:
    """The width over height a ``W:H`` enum value names, or None for a word such as ``adaptive``."""
    width, separator, height = name.partition(":")
    if not separator or not width.isdecimal() or not height.isdecimal() or int(width) == 0 or int(height) == 0:
        return None
    return int(width) / int(height)


class VideoSchema:
    """The input schema of one DeepInfra video model, read from ``GET /models/{model}``."""

    def __init__(self, model: str, in_schema: dict[str, Any]) -> None:
        self.model = model
        self.fields: dict[str, dict[str, Any]] = in_schema.get("properties") or {}
        self.definitions: dict[str, dict[str, Any]] = in_schema.get("definitions") or in_schema.get("$defs") or {}

    @classmethod
    def from_model_record(cls, model: str, record: dict[str, Any]) -> VideoSchema:
        """Read the schema out of a ``GET /models/{model}`` reply, or fail when the model has none."""
        in_schema = record.get("in_schema")
        if not isinstance(in_schema, dict) or not isinstance(in_schema.get("properties"), dict):
            msg = f"DeepInfra published no input schema for {model}, so it cannot be used for video generation"
            raise ProviderError(msg)
        return cls(model, in_schema)

    def enum(self, field: str) -> list[str] | None:
        """The allowed values of a string field, or None when the field is missing or free-form."""
        values = (self.fields.get(field) or {}).get("enum")
        return [str(value) for value in values] if values else None

    def media_types(self) -> list[str]:
        """The item types a ``media`` array takes, or an empty list when the model has no such field."""
        items = (self.fields.get("media") or {}).get("items") or {}
        ref = str(items.get("$ref") or "")
        definition = self.definitions.get(ref.rsplit("/", 1)[-1], items)
        return [str(value) for value in ((definition.get("properties") or {}).get("type") or {}).get("enum") or []]


class _BodyBuilder:
    def __init__(self, params: VideoGenerationParams, schema: VideoSchema, provider_name: str) -> None:
        self.params = params
        self.schema = schema
        self.provider_name = provider_name
        self.body: dict[str, Any] = {"prompt": params.prompt}
        self.media: list[dict[str, Any]] = []

    def reject(self, parameter_name: str) -> NoReturn:
        """Raise for an any-llm field this model's schema has no name for."""
        hint = (
            f"{self.schema.model} on DeepInfra has no field for it; pass the model's own field as a keyword argument."
        )
        raise UnsupportedParameterError(parameter_name, self.provider_name, hint)

    def plain(self, parameter_name: str, field: str, value: Any) -> None:
        """A field the schema has under one fixed name."""
        if field not in self.schema.fields:
            self.reject(parameter_name)
        self.body[field] = value

    def switch(self, parameter_name: str, field_names: tuple[str, ...], value: bool, kwargs: dict[str, Any]) -> None:
        """A yes or no the model takes as one of its boolean fields, unless ``kwargs`` set that field the other way."""
        field = next(
            (name for name in field_names if (self.schema.fields.get(name) or {}).get("type") == "boolean"), None
        )
        if field is None:
            self.reject(parameter_name)
        # A gateway bills on the shared field, so the request must not render the opposite.
        if field in kwargs and kwargs[field] != value:
            msg = f"{parameter_name}={value} conflicts with {field}={kwargs[field]!r}; pass only one of them"
            raise InvalidRequestError(msg)
        self.body[field] = value

    def image(self, parameter_name: str, value: str, field_names: tuple[str, ...], media_type: str) -> None:
        """An image as the model's own field, or as a typed ``media`` item when that is how it takes images."""
        field = next((name for name in field_names if name in self.schema.fields), None)
        if field is not None:
            self.body[field] = value
        elif media_type in self.schema.media_types():
            self.media.append({"type": media_type, "url": value})
        else:
            self.reject(parameter_name)

    def references(self, parameter_name: str, urls: list[str], media_type: str) -> None:
        """References as the model's list field of the same name, or as typed ``media`` items."""
        if parameter_name in self.schema.fields:
            self.body[parameter_name] = list(urls)
        elif media_type in self.schema.media_types():
            self.media.extend({"type": media_type, "url": url} for url in urls)
        else:
            self.reject(parameter_name)

    def reference_audios(self, audios: list[str]) -> None:
        """Audio as the model's list field, as ``media`` items, or as its single-clip field when there is one clip."""
        if _REFERENCE_AUDIOS_FIELD in self.schema.fields:
            self.body[_REFERENCE_AUDIOS_FIELD] = list(audios)
        elif "reference_audio" in self.schema.media_types():
            self.media.extend({"type": "reference_audio", "url": audio} for audio in audios)
        elif (field := next((name for name in _SINGLE_AUDIO_FIELDS if name in self.schema.fields), None)) is not None:
            if len(audios) != 1:
                msg = f"{self.schema.model} takes one audio clip as {field}, not {len(audios)}"
                raise InvalidRequestError(msg)
            self.body[field] = audios[0]
        else:
            self.reject("reference_audios")

    def shape(self) -> None:
        """The shape as the model's resolution tier and ratio, or as its exact ``WIDTH*HEIGHT`` size."""
        shape = self.params.parse_shape()
        if shape is None:
            return
        if (sizes := self.schema.enum("size")) is not None:
            # Only an exact size names one of these; a tier and ratio would be a guess.
            if shape.dimensions is None:
                msg = f"{self.schema.model} takes an exact size ({', '.join(sizes)}), so pass size, not {shape.described!r}"
                raise InvalidRequestError(msg)
            width, height = shape.dimensions
            if f"{width}*{height}" not in sizes:
                msg = f"{self.schema.model} renders {', '.join(sizes)}, not {shape.described!r}"
                raise InvalidRequestError(msg)
            self.body["size"] = f"{width}*{height}"
            return
        if shape.short_side is not None:
            self.resolution_tier(shape.short_side, shape.described)
        if shape.ratio is not None:
            self.ratio(shape.ratio, shape.described)

    def resolution_tier(self, short_side: int, described: str) -> None:
        """The short side as the model's own spelling of that tier, ``720p`` or ``720P``."""
        resolutions = self.schema.enum("resolution")
        if resolutions is None:
            self.reject("size" if self.params.size is not None else "resolution")
        resolution = next((value for value in resolutions if value.lower() == f"{short_side}p"), None)
        if resolution is None:
            msg = f"{self.schema.model} renders video with a resolution of {', '.join(resolutions)}, not {described!r}"
            raise InvalidRequestError(msg)
        self.body["resolution"] = resolution

    def ratio(self, ratio: float, described: str) -> None:
        """The width over height as the nearest ``W:H`` value the model lists."""
        ratio_field = next((name for name in _RATIO_FIELDS if name in self.schema.fields), None)
        if ratio_field is None:
            if self.params.aspect_ratio is not None:
                self.reject("aspect_ratio")
            msg = (
                f"{self.schema.model} takes a resolution tier but no shape, so pass resolution, not size {described!r}"
            )
            raise InvalidRequestError(msg)
        ratios = {
            name: value for name in self.schema.enum(ratio_field) or [] if (value := _ratio_value(name)) is not None
        }
        nearest = min(ratios, key=lambda name: abs(ratios[name] - ratio), default=None)
        if nearest is None or abs(ratios[nearest] - ratio) / ratio > _ASPECT_RATIO_TOLERANCE:
            msg = f"{self.schema.model} renders video with an aspect ratio of {', '.join(ratios)}, not {described!r}"
            raise InvalidRequestError(msg)
        self.body[ratio_field] = nearest

    def build(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """The request body: every set any-llm field under the model's name, then ``kwargs`` as is."""
        params = self.params
        if params.seconds is not None:
            self.plain("seconds", "duration", params.seconds)
        if params.seed is not None:
            self.plain("seed", "seed", params.seed)
        if params.negative_prompt is not None:
            self.plain("negative_prompt", "negative_prompt", params.negative_prompt)
        self.shape()
        for parameter_name, field_names in _FRAME_FIELDS.items():
            if (value := getattr(params, parameter_name)) is not None:
                self.image(parameter_name, value, field_names, parameter_name)
        if params.reference_images:
            self.references("reference_images", params.reference_images, "reference_image")
        if params.reference_audios:
            self.reference_audios(params.reference_audios)
        if params.reference_videos:
            self.references("reference_videos", params.reference_videos, "reference_video")
        if params.generate_audio is not None:
            self.switch("generate_audio", _GENERATE_AUDIO_FIELDS, params.generate_audio, kwargs)
        # Caller-provided media items merge with the ones any-llm's fields produced.
        self.media.extend(kwargs.pop("media", None) or [])
        if self.media:
            self.body["media"] = self.media
        return self.body | kwargs


def convert_video_params(
    params: VideoGenerationParams, schema: VideoSchema, provider_name: str, **kwargs: Any
) -> dict[str, Any]:
    """Translate VideoGenerationParams, plus pass-through kwargs, into the body of ``POST /v1/inference/{model}``.

    Each any-llm field goes in under whatever name the model's schema gives it. A field the
    schema has no name for raises ``UnsupportedParameterError``; keyword arguments go in as is.
    """
    return _BodyBuilder(params, schema, provider_name).build(kwargs)


def _absolute_url(url: str, api_base: str) -> str:
    """DeepInfra documents ``video_url`` as a path in its example, so resolve one against the API host."""
    if url.startswith(("http://", "https://")):
        return url
    scheme, _, rest = api_base.partition("://")
    host = rest.split("/", 1)[0]
    return f"{scheme}://{host}/{url.lstrip('/')}"


def convert_inference_response(response: dict[str, Any], *, model: str, api_base: str) -> VideoJob:
    """Read a synchronous inference reply as a finished VideoJob.

    DeepInfra renders inside the request and returns the video URL in the same reply, so the
    job is already ``completed`` (or ``failed``) when it is first seen.
    """
    inference_status = response.get("inference_status") or {}
    raw_status = str(inference_status.get("status") or "unknown")
    status = _STATUS_MAP.get(raw_status)
    if status is None:
        msg = f"DeepInfra returned an unknown inference status {raw_status!r}"
        raise ProviderError(msg)

    video_url = response.get("video_url")
    outputs: list[VideoOutput] = []
    if status == "completed":
        if video_url:
            outputs.append(VideoOutput(url=_absolute_url(str(video_url), api_base)))
        else:
            # A reply with no video and no failure status has nothing to download, which is a failure
            # for the caller whatever DeepInfra called it.
            status = "failed"

    error: VideoError | None = None
    if status == "failed":
        detail = response.get("detail") or response.get("error") or "DeepInfra returned no video."
        error = VideoError(message=str(detail))

    return VideoJob(
        id=str(response.get("request_id") or f"deepinfra-{uuid4().hex}"),
        model=model,
        status=status,
        error=error,
        outputs=outputs,
    )
