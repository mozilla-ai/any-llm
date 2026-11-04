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
    """``WIDTHxHEIGHT``, such as ``1280x720``."""
    seed: int | None = None
    """Seed for reproducible generation."""
    negative_prompt: str | None = None
    """What the video should not contain."""
    reference_images: list[str] | None = None
    """Image URLs, provider file URIs or ``data:`` URLs the video takes its subjects or style from."""
    first_frame: str | None = None
    """Image the video starts from, in the same forms as `reference_images`."""
    last_frame: str | None = None
    """Image the video ends on, in the same forms as `reference_images`."""

    def to_api_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for the provider API call, excluding None values."""
        return {k: v for k, v in self.model_dump(exclude={"model_id"}).items() if v is not None}

    def parse_size(self) -> tuple[int, int] | None:
        """Return ``(width, height)`` from ``size``, or ``None`` when no size was given.

        Raises:
            InvalidRequestError: If ``size`` is not ``WIDTHxHEIGHT``.

        """
        if self.size is None:
            return None
        width, separator, height = self.size.lower().partition("x")
        if not separator or not width.isdigit() or not height.isdigit():
            msg = f"size must be WIDTHxHEIGHT, such as 1280x720, not {self.size!r}"
            raise InvalidRequestError(msg)
        return int(width), int(height)
