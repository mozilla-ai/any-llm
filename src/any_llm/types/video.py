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
    code: str | None = None


class VideoOutput(BaseModel):
    """One generated video, as a URL, inline base64 data, or both."""

    url: str | None = None
    b64_data: str | None = None
    mime_type: str = "video/mp4"


class VideoJob(BaseModel):
    """A video generation job.

    Generation is slow, so every provider returns a job first. Poll it with
    ``retrieve_video`` until ``done``; ``outputs`` is empty until then.
    """

    id: str
    model: str
    status: VideoStatus
    created_at: int | None = None
    completed_at: int | None = None
    progress: int | None = None
    """Percent complete, when the provider reports it."""
    seconds: int | None = None
    size: str | None = None
    """``WIDTHxHEIGHT`` of the output, when the provider reports it."""
    error: VideoError | None = None
    outputs: list[VideoOutput] = Field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.status in TERMINAL_VIDEO_STATUSES


class VideoGenerationParams(BaseModel):
    """Parameters for video generation requests."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    prompt: str
    seconds: int | None = None
    size: str | None = None
    seed: int | None = None
    negative_prompt: str | None = None
    reference_images: list[str] | None = None
    first_frame: str | None = None
    last_frame: str | None = None

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
