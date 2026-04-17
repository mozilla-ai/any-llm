"""Types for the moderation API.

Mirrors the OpenAI moderation response shape while allowing providers to
populate only the categories they actually return. Absent keys mean
"unknown" for that provider; callers should not assume zero-filled
defaults.
"""

from typing import Any

from pydantic import BaseModel, Field


class ModerationResult(BaseModel):
    """A single moderation decision, typically one per input item."""

    flagged: bool
    categories: dict[str, bool] = Field(default_factory=dict)
    category_scores: dict[str, float] = Field(default_factory=dict)
    category_applied_input_types: dict[str, list[str]] | None = None
    provider_raw: dict[str, Any] | None = None


class ModerationResponse(BaseModel):
    """Normalized moderation response across providers."""

    id: str
    model: str
    results: list[ModerationResult]
