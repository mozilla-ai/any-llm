"""OpenAI-compatible embeddings endpoint."""

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from any_llm import AnyLLM, aembedding
from any_llm.gateway.api.deps import get_config, get_db, verify_api_key_or_master_key
from any_llm.gateway.api.routes.chat import get_provider_kwargs, rate_limit_headers
from any_llm.gateway.core.config import GatewayConfig
from any_llm.gateway.log_config import logger
from any_llm.gateway.models.entities import APIKey, UsageLog, User
from any_llm.gateway.rate_limit import check_rate_limit
from any_llm.gateway.services.budget_service import validate_user_budget
from any_llm.gateway.services.pricing_service import find_model_pricing
from any_llm.types.completion import CreateEmbeddingResponse

router = APIRouter(prefix="/v1", tags=["embeddings"])


class EmbeddingRequest(BaseModel):
    """OpenAI-compatible embedding request."""

    model: str
    input: str | list[str] = Field(description="Input text to embed")
    user: str | None = None
    encoding_format: str | None = None
    dimensions: int | None = None


@router.post("/embeddings", response_model=None)
async def create_embedding(
    raw_request: Request,
    response: Response,
    request: EmbeddingRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> CreateEmbeddingResponse:
    """OpenAI-compatible embeddings endpoint.

    Authentication modes:
    - Master key + user field: Use specified user (must exist)
    - API key + user field: Use specified user (must exist)
    - API key without user field: Use virtual user created with API key
    """
    api_key, is_master_key = auth_result

    user_id: str
    if is_master_key:
        if not request.user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="When using master key, 'user' field is required in request body",
            )
        user_id = request.user
    elif request.user:
        user_id = request.user
    else:
        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API key validation failed",
            )
        if not api_key.user_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API key has no associated user",
            )
        user_id = str(api_key.user_id)

    rate_limit_info = check_rate_limit(raw_request, user_id)

    _ = await validate_user_budget(db, user_id, request.model, strategy=config.budget_strategy)

    provider, model = AnyLLM.split_model_provider(request.model)

    provider_kwargs = get_provider_kwargs(config, provider)

    embedding_kwargs: dict[str, Any] = {
        "model": model,
        "inputs": request.input,
        "provider": provider,
        **provider_kwargs,
    }
    if request.encoding_format is not None:
        embedding_kwargs["encoding_format"] = request.encoding_format
    if request.dimensions is not None:
        embedding_kwargs["dimensions"] = request.dimensions

    try:
        result = await aembedding(**embedding_kwargs)

        usage_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key.id if api_key else None,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=provider,
            endpoint="/v1/embeddings",
            status="success",
            prompt_tokens=result.usage.prompt_tokens if result.usage else None,
            completion_tokens=0,
            total_tokens=result.usage.total_tokens if result.usage else None,
        )

        if result.usage:
            pricing = await find_model_pricing(db, provider, model)
            if pricing:
                cost = (result.usage.prompt_tokens / 1_000_000) * pricing.input_price_per_million
                usage_log.cost = cost

                await db.execute(
                    update(User)
                    .where(User.user_id == user_id, User.deleted_at.is_(None))
                    .values(spend=User.spend + cost)
                )
            else:
                model_ref = f"{provider}:{model}" if provider else model
                logger.warning(f"No pricing configured for '{model_ref}'. Usage will be tracked without cost.")

        try:
            db.add(usage_log)
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to log usage to database: {e}")
            await db.rollback()

    except HTTPException:
        raise
    except Exception as e:
        error_log = UsageLog(
            id=str(uuid.uuid4()),
            api_key_id=api_key.id if api_key else None,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            model=model,
            provider=provider,
            endpoint="/v1/embeddings",
            status="error",
            error_message=str(e),
        )
        try:
            db.add(error_log)
            await db.commit()
        except Exception as log_err:
            logger.error(f"Failed to log usage to database: {log_err}")
            await db.rollback()

        logger.error(f"Provider call failed for {provider}:{model}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The request could not be completed by the provider",
        ) from e

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    return result
