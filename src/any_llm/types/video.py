"""Video generation types for any_llm."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from any_llm.exceptions import InvalidRequestError

VideoStatus = Literal["queued", "in_progress", "completed", "failed"]
"""Lifecycle of a video job. Providers' own words (``pending``, ``processing``, ``success``) map onto these."""

TERMINAL_VIDEO_STATUSES: frozenset[str] = frozenset({"completed", "failed"})


class VideoError(BaseModel):
    """Why a video job failed."""

    message: str
    """What went wrong, in the provider's words."""
    code: str | None = None
    """The provider's error code, when it gives one."""


class VideoOutput(BaseModel):
    """One generated video, as a URL, inline base64 data, or both."""

    url: str | None = None
    """Where the video can be fetched. Some providers' URLs need their credentials; `download_video` adds them."""
    b64_data: str | None = None
    """The video, base64-encoded, when the provider returns it inline."""
    mime_type: str = "video/mp4"
    """Media type of the video."""


class VideoJob(BaseModel):
    """A video generation job.

    Generation is slow, so every provider returns a job first. Poll it with
    ``retrieve_video`` until ``done``; ``outputs`` is empty until then.
    """

    id: str
    """Provider-assigned job ID; pass it to `retrieve_video`."""
    model: str
    """Model that renders the video."""
    status: VideoStatus
    """Where the job is. Providers' own words map onto these four."""
    created_at: int | None = None
    """Unix timestamp of the request, when the provider reports it."""
    completed_at: int | None = None
    """Unix timestamp of completion, when the provider reports it."""
    progress: int | None = None
    """Percent complete, when the provider reports it."""
    seconds: int | None = None
    """Length of the video in seconds, when the provider reports it."""
    size: str | None = None
    """``WIDTHxHEIGHT`` of the output, when the provider reports it."""
    error: VideoError | None = None
    """Why a `failed` job failed."""
    outputs: list[VideoOutput] = Field(default_factory=list)
    """The rendered videos; empty until `completed`."""

    @property
    def done(self) -> bool:
        return self.status in TERMINAL_VIDEO_STATUSES


class VideoShape(BaseModel):
    """What a video should look like, read from ``size`` or from ``resolution`` and ``aspect_ratio``."""

    short_side: int | None
    """Pixels on the short side, which names the resolution tier (720 for ``720p``)."""
    ratio: float | None
    """Width over height, so 16:9 is about 1.78 and a portrait video is below 1."""
    dimensions: tuple[int, int] | None
    """Exact ``(width, height)``, only when the shape came from ``size``."""
    described: str
    """What the caller wrote, for error messages."""


class VideoGenerationParams(BaseModel):
    """Parameters for video generation requests."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    """The provider's own model ID, without a ``provider:`` prefix."""
    prompt: str
    """What the video shows."""
    seconds: int | None = None
    """Length of the video in seconds."""
    size: str | None = None
    """``WIDTHxHEIGHT``, such as ``1280x720``. Says both how sharp and what shape; not combined with
    `resolution` or `aspect_ratio`."""
    resolution: str | None = None
    """How sharp, by short side: ``480p``, ``720p``, ``1080p`` or ``4k``. Pair it with `aspect_ratio` for the shape."""
    aspect_ratio: str | None = None
    """What shape, as ``WIDTH:HEIGHT`` such as ``16:9`` or ``9:16``."""
    seed: int | None = None
    """Seed for reproducible generation."""
    negative_prompt: str | None = None
    """What the video should not contain."""
    reference_images: list[str] | None = None
    """Image URLs, provider file URIs or ``data:`` URLs the video takes its subjects or style from."""
    reference_audios: list[str] | None = None
    """Audio URLs the video follows, for lip sync, beat-matched motion or a sound track."""
    reference_videos: list[str] | None = None
    """Video URLs or provider file URIs the video takes its motion, subjects or style from."""
    first_frame: str | None = None
    """Image the video starts from, in the same forms as `reference_images`."""
    last_frame: str | None = None
    """Image the video ends on, in the same forms as `reference_images`."""
    generate_audio: bool | None = None
    """Whether the video carries a sound track. ``None`` leaves the provider's default."""

    def to_api_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for the provider API call, excluding None values."""
        return {k: v for k, v in self.model_dump(exclude={"model_id"}).items() if v is not None}

    def merge_provider_options(self, fields: dict[str, tuple[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        """Return the request fields the shared parameters set, with provider ``options`` laid over them.

        A gateway prices a render from the shared parameters, so an option may repeat a field one
        of them set but not change it.

        Args:
            fields: Each request field a shared parameter set, mapped to that parameter's name and the value it set.
            options: Provider options for the same level of the request.

        Raises:
            InvalidRequestError: If an option sets one of ``fields`` to another value.

        """
        self.check_provider_options(fields, options)
        return {field: value for field, (_, value) in fields.items()} | options

    def check_provider_options(self, fields: dict[str, tuple[str, Any]], options: dict[str, Any]) -> None:
        """Raise when provider ``options`` set one of ``fields`` to another value than its shared parameter did.

        Args:
            fields: Each request field a shared parameter set, mapped to that parameter's name and the value it set.
            options: Provider options that can reach the same request fields.

        Raises:
            InvalidRequestError: If an option sets one of ``fields`` to another value.

        """
        for field, (parameter_name, value) in fields.items():
            if field in options and options[field] != value:
                shared = getattr(self, parameter_name)
                msg = f"{parameter_name}={shared!r} conflicts with {field}={options[field]!r}; pass only one of them"
                raise InvalidRequestError(msg)

    def parse_size(self) -> tuple[int, int] | None:
        """Return ``(width, height)`` from ``size``, or ``None`` when no size was given.

        Raises:
            InvalidRequestError: If ``size`` is not ``WIDTHxHEIGHT``.

        """
        if self.size is None:
            return None
        width, separator, height = self.size.lower().partition("x")
        if not separator or not width.isdecimal() or not height.isdecimal() or int(width) == 0 or int(height) == 0:
            msg = f"size must be WIDTHxHEIGHT, such as 1280x720, not {self.size!r}"
            raise InvalidRequestError(msg)
        return int(width), int(height)

    def parse_shape(self) -> VideoShape | None:
        """Return the requested shape from ``size``, or from ``resolution`` and ``aspect_ratio``.

        Returns:
            None when none of the three was given.

        Raises:
            InvalidRequestError: If ``size`` is combined with the other two, or any of them is malformed.

        """
        if self.size is not None:
            if self.resolution is not None or self.aspect_ratio is not None:
                msg = "Pass either size or resolution and aspect_ratio, not both."
                raise InvalidRequestError(msg)
            width, height = self.parse_size()  # type: ignore[misc]
            return VideoShape(
                short_side=min(width, height), ratio=width / height, dimensions=(width, height), described=self.size
            )
        if self.resolution is None and self.aspect_ratio is None:
            return None
        short_side: int | None = None
        if self.resolution is not None:
            tier = self.resolution.lower()
            if tier == "4k":
                short_side = 2160
            elif tier.endswith("p") and tier[:-1].isdecimal():
                short_side = int(tier[:-1])
            else:
                msg = f"resolution must be a short side such as 720p, or 4k, not {self.resolution!r}"
                raise InvalidRequestError(msg)
        ratio: float | None = None
        if self.aspect_ratio is not None:
            numerator, separator, denominator = self.aspect_ratio.partition(":")
            if (
                not separator
                or not numerator.isdecimal()
                or not denominator.isdecimal()
                or int(numerator) == 0
                or int(denominator) == 0
            ):
                msg = f"aspect_ratio must be WIDTH:HEIGHT, such as 16:9, not {self.aspect_ratio!r}"
                raise InvalidRequestError(msg)
            ratio = int(numerator) / int(denominator)
        described = " ".join(value for value in (self.resolution, self.aspect_ratio) if value is not None)
        return VideoShape(short_side=short_side, ratio=ratio, dimensions=None, described=described)
