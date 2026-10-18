from openai.types import CreateEmbeddingResponse
from openai.types.create_embedding_response import Usage
from openai.types.embedding import Embedding
from voyageai.object.embeddings import EmbeddingsObject
from voyageai.object.reranking import RerankingObject

from any_llm.types.rerank import RerankResponse, RerankResult, RerankUsage


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


def _convert_voyage_rerank_response(result: RerankingObject) -> RerankResponse:
    """Convert a Voyage AI rerank response to a normalized RerankResponse."""

    results = [
        RerankResult(
            index=r.index,
            relevance_score=r.relevance_score,
        )
        for r in result.results
    ]
    # Defensive: Voyage returns sorted but re-sort to guarantee the docstring contract
    results.sort(key=lambda r: r.relevance_score, reverse=True)

    # Voyage does not return a response ID, so RerankResponse.id stays None.
    return RerankResponse(
        results=results,
        usage=RerankUsage(total_tokens=result.total_tokens),
    )
