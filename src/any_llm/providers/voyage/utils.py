from openai.types import CreateEmbeddingResponse
from openai.types.create_embedding_response import Usage
from openai.types.embedding import Embedding
from voyageai.object.embeddings import EmbeddingsObject


def _create_openai_embedding_response_from_voyage(model: str, result: EmbeddingsObject) -> CreateEmbeddingResponse:
    """Convert a Voyage AI embedding response to an OpenAI-compatible format."""

    data = [
        Embedding(
            embedding=embedding,  # type: ignore[arg-type]
            index=i,
            object="embedding",
        )
        for i, embedding in enumerate(result.embeddings or [])
    ]

    usage = Usage(prompt_tokens=result.total_tokens, total_tokens=result.total_tokens)

    return CreateEmbeddingResponse(
        data=data,
        model=model,
        object="list",
        usage=usage,
    )
