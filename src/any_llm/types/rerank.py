from __future__ import annotations

from pydantic import BaseModel, Field


class RerankResult(BaseModel):
    """A single document's relevance score."""

    index: int = Field(description="Zero-based index into the original documents list")
    relevance_score: float = Field(description="Relevance score, higher is more relevant")


class RerankMeta(BaseModel):
    """Provider-specific billing metadata (optional, preserved as-is)."""

    billed_units: dict[str, float] | None = None
    tokens: dict[str, int] | None = None


class RerankUsage(BaseModel):
    """Normalized token usage for gateway logging."""

    total_tokens: int | None = None


class RerankResponse(BaseModel):
    """Normalized rerank response, provider-agnostic."""

    id: str = Field(description="Provider-assigned response ID")
    results: list[RerankResult] = Field(description="Results sorted by relevance_score descending")
    meta: RerankMeta | None = None
    usage: RerankUsage | None = None
