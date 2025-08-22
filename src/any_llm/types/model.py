from typing import Any

from openai.types.model import Model as OpenAIModel
from pydantic import Field


class Model(OpenAIModel):
    label: str | None = None
    provider: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
