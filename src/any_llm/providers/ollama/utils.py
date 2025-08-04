from typing import Any

from openai.types import CreateEmbeddingResponse
from openai.types.embedding import Embedding
from openai.types.create_embedding_response import Usage


def _create_openai_embedding_response_from_ollama(
    ollama_response: Any,
) -> CreateEmbeddingResponse:
    """Convert an Ollama embedding response to OpenAI CreateEmbeddingResponse format."""

    openai_embeddings = []

    for index, embedding_vector in enumerate(ollama_response.get("embeddings", [])):
        openai_embedding = Embedding(embedding=embedding_vector, index=index, object="embedding")
        openai_embeddings.append(openai_embedding)

    prompt_tokens = ollama_response.get("prompt_eval_count", 0)
    usage = Usage(
        prompt_tokens=prompt_tokens,
        total_tokens=prompt_tokens,
    )

    return CreateEmbeddingResponse(
        data=openai_embeddings,
        model=ollama_response.get("model", "unknown"),
        object="list",
        usage=usage,
    )
