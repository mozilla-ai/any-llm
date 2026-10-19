"""Image generation and edit types for any_llm."""

from typing import Any, Literal, TypeAlias

from openai._types import FileTypes
from openai.types import ImagesResponse as OpenAIImagesResponse
from openai.types.image import Image as OpenAIImage
from pydantic import BaseModel, ConfigDict, SkipValidation

ImageInput: TypeAlias = FileTypes | list[FileTypes]
"""One image or a list of them, each as bytes, a binary file object, a ``pathlib.Path``, or a
``(filename, content, content_type)`` tuple."""


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


class ImageEditParams(BaseModel):
    """Parameters for image edit requests."""

    # The images go to the provider SDK as given; pydantic cannot check file objects.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model_id: str
    image: SkipValidation[ImageInput]
    prompt: str
    n: int | None = None
    size: str | None = None
    quality: str | None = None
    response_format: Literal["url", "b64_json"] | None = None
    user: str | None = None

    def to_api_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for the provider API call, excluding None values and the images."""
        return {k: v for k, v in self.model_dump(exclude={"model_id", "image"}).items() if v is not None}


# Re-export OpenAI types for convenience
ImagesResponse = OpenAIImagesResponse
Image = OpenAIImage
