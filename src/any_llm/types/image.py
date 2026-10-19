"""Image generation and edit types for any_llm."""

from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

from openai._types import FileTypes
from openai.types import ImagesResponse as OpenAIImagesResponse
from openai.types.image import Image as OpenAIImage
from pydantic import BaseModel, ConfigDict, PlainValidator

from any_llm.exceptions import InvalidRequestError

ImageFile: TypeAlias = FileTypes | str
"""One image: bytes, a path, a named binary file object, or a ``(filename, content, content_type)`` tuple."""

ImageInput: TypeAlias = ImageFile | list[ImageFile]
"""One image or a list of them."""

# Fallback name for image bytes that arrive without one. Multipart file parts need a filename.
DEFAULT_IMAGE_FILENAME = "image"

# Signatures of the formats the image edit endpoints accept, so bytes that arrive without a name
# still go up with a content type the provider recognizes instead of application/octet-stream.
_IMAGE_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (b"\xff\xd8\xff", "jpg", "image/jpeg"),
)

_TUPLE_LENGTHS = range(2, 5)


def sniff_image_format(content: bytes) -> tuple[str, str] | None:
    """Return ``(extension, content_type)`` for PNG, JPEG or WEBP bytes, else ``None``."""
    for magic, extension, content_type in _IMAGE_SIGNATURES:
        if content.startswith(magic):
            return extension, content_type
    if content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "webp", "image/webp"
    return None


def _is_file_content(value: Any) -> bool:
    # typing.IO is not usable with isinstance, so accept anything readable instead.
    return isinstance(value, (bytes, PathLike)) or callable(getattr(value, "read", None))


def _validate_image_file(value: Any) -> Any:
    if isinstance(value, str) or _is_file_content(value):
        return value
    # A str inside a tuple is uploaded by httpx as the literal file content, not read as a path.
    if (
        isinstance(value, tuple)
        and len(value) in _TUPLE_LENGTHS
        and (value[0] is None or isinstance(value[0], str))
        and _is_file_content(value[1])
    ):
        return value
    msg = "image must be bytes, a path, a readable binary file object, or a (filename, content, content_type) tuple"
    raise ValueError(msg)


def _validate_image_input(value: Any) -> Any:
    """Accept one image or a non-empty list of them."""
    if isinstance(value, list):
        if not value:
            msg = "image must contain at least one image"
            raise ValueError(msg)
        return [_validate_image_file(item) for item in value]
    return _validate_image_file(value)


def _resolve_image(image: ImageFile) -> FileTypes:
    if isinstance(image, (str, PathLike)):
        path = Path(image)
        try:
            content = path.read_bytes()
        except (OSError, ValueError) as exc:
            # Without this, exception conversion classifies FileNotFoundError by its type name as
            # ModelNotFoundError. ValueError is what an embedded NUL byte in the path raises.
            msg = f"Cannot read image path {str(path)!r}: {getattr(exc, 'strerror', None) or exc}"
            raise InvalidRequestError(msg, original_exception=exc) from exc
        sniffed = sniff_image_format(content)
        return (path.name, content, sniffed[1] if sniffed else None)
    if isinstance(image, bytes):
        sniffed = sniff_image_format(image)
        if sniffed is None:
            return image
        return (f"{DEFAULT_IMAGE_FILENAME}.{sniffed[0]}", image, sniffed[1])
    return image


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

    # The validator checks the shape of each image; pydantic cannot check or copy file objects.
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    model_id: str
    image: Annotated[ImageInput, PlainValidator(_validate_image_input)]
    prompt: str
    n: int | None = None
    size: str | None = None
    quality: str | None = None
    response_format: Literal["url", "b64_json"] | None = None
    user: str | None = None

    def to_api_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs for the provider API call, excluding None values and the images."""
        return {k: v for k, v in self.model_dump(exclude={"model_id", "image"}).items() if v is not None}

    def resolve_images(self) -> FileTypes | list[FileTypes]:
        """Return ``image`` in the shape the OpenAI SDK uploads, keeping a single image single.

        Paths are read here so that one that cannot be read is reported as a bad request. Bytes
        in a recognized format get a name and content type, because OpenAI rejects image parts
        sent as application/octet-stream.
        """
        if isinstance(self.image, list):
            return [_resolve_image(item) for item in self.image]
        return _resolve_image(self.image)


# Re-export OpenAI types for convenience
ImagesResponse = OpenAIImagesResponse
Image = OpenAIImage
