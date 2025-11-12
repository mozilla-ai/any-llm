import re

from pydantic import BaseModel, field_validator

ANY_API_KEY_REGEX = r"^ANY\.v\d+\.([^.]+)\.([^-]+)-(.+)$"


class ProviderMetadata(BaseModel):
    name: str
    env_key: str
    doc_url: str
    streaming: bool
    reasoning: bool
    completion: bool
    embedding: bool
    responses: bool
    image: bool
    pdf: bool
    class_name: str
    list_models: bool
    batch_completion: bool


class PlatformKey(BaseModel):
    api_key: str

    @field_validator("api_key")
    @classmethod
    def validate_api_key_format(cls, value: str) -> str:
        """Validates the API key against the required format."""
        if not re.fullmatch(ANY_API_KEY_REGEX, value):
            msg = "Invalid API key format. Must match the pattern: ANY.<version>.<kid>.<fingerprint>-<base64_key>."
            raise ValueError(msg)
        return value
