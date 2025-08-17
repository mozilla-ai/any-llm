from pydantic import BaseModel


class ModelMetadata(BaseModel):
    id: str
    created: int | None = None
    object: str | None = None
