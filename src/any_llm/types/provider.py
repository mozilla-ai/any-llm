from pydantic import BaseModel

from any_llm.constants import ProviderTier


class ProviderMetadata(BaseModel):
    name: str
    tier: ProviderTier = ProviderTier.COMMUNITY
    """Support tier. Defaults to community, so a provider is only advertised as
    verified once it is listed in VERIFIED_PROVIDERS."""
    env_key: str
    env_api_base: str | None
    doc_url: str
    streaming: bool
    reasoning: bool
    completion: bool
    embedding: bool
    moderation: bool
    responses: bool
    image: bool
    pdf: bool
    class_name: str
    list_models: bool
    messages: bool
    batch_completion: bool
    image_generation: bool = False
    audio_transcription: bool = False
    audio_speech: bool = False
    rerank: bool = False
