"""Image generation types for any_llm."""

from typing import Any, Literal

from openai.types import ImagesResponse as OpenAIImagesResponse
from openai.types.image import Image as OpenAIImage
from pydantic import BaseModel, ConfigDict


class ImageGenerationParams(BaseModel):
    """Parameters for image generation requests."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    prompt: str
    n: int | None = None
    size: str | None = None
    quality: str | None = None
    style: str | None = None
    response_format: Literal["url", "b64_json"] | None = None
    user: str | None = None

    def to_api_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for the provider API call, excluding None values."""
        return {k: v for k, v in self.model_dump(exclude={"model_id"}).items() if v is not None}


# Re-export OpenAI types for convenience
ImagesResponse = OpenAIImagesResponse
Image = OpenAIImage
