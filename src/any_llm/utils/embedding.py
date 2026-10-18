"""OpenAI ``encoding_format`` support for providers whose embedding API returns floats only."""

from __future__ import annotations

import base64
import struct
from typing import TYPE_CHECKING, Any, Literal, cast

from any_llm.exceptions import UnsupportedParameterError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from any_llm.types.completion import CreateEmbeddingResponse

EncodingFormat = Literal["float", "base64"]


def pop_encoding_format(kwargs: dict[str, Any], provider_name: str) -> EncodingFormat | None:
    """Remove ``encoding_format`` from ``kwargs`` so it never reaches a provider SDK that rejects it.

    The OpenAI SDK sends ``encoding_format="base64"`` on every ``embeddings.create`` call by default,
    so OpenAI-compatible callers routinely pass it. Pass the returned value to
    `apply_encoding_format` once the provider has returned its float vectors.
    """
    encoding_format = kwargs.pop("encoding_format", None)
    if encoding_format is None:
        return None
    if encoding_format in ("float", "base64"):
        return cast("EncodingFormat", encoding_format)
    parameter_name = f"encoding_format={encoding_format!r}"
    raise UnsupportedParameterError(parameter_name, provider_name, "Use 'float' or 'base64'.")


def apply_encoding_format(
    response: CreateEmbeddingResponse, encoding_format: EncodingFormat | None
) -> CreateEmbeddingResponse:
    """Encode each embedding the way OpenAI does for ``base64``: little-endian float32 bytes, base64."""
    if encoding_format != "base64":
        return response
    data = [item.model_copy(update={"embedding": _encode_vector(item.embedding)}) for item in response.data]
    return response.model_copy(update={"data": data})


def _encode_vector(vector: Sequence[float]) -> str:
    return base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")
