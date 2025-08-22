from pydantic import BaseModel


class ProviderMetadata(BaseModel):
    name: str
    label: str
    env_key: str
    doc_url: str
    streaming: bool
    reasoning: bool
    completion: bool
    embedding: bool
    responses: bool
    class_name: str
    list_models: bool
