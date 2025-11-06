import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from any_llm import AnyLLM, LLMProvider, acompletion
from any_llm.gateway.auth import verify_api_key_or_master_key
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.auth.vertex_auth import setup_vertex_environment
from any_llm.gateway.budget import validate_user_budget
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, ModelPricing, UsageLog, User, get_db
from any_llm.gateway.log_config import logger
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionUsage

router = APIRouter(prefix="/v1/chat", tags=["chat"])


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[dict[str, Any]]
    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None


def _get_provider_credentials(
    config: GatewayConfig,
    provider: LLMProvider,
) -> dict[str, Any]:
    """Get provider credentials from config.

    Args:
        config: Gateway configuration
        provider: Provider name

    Returns:
        Dictionary of provider credentials

    Raises:
        HTTPException: If credentials not found

    """
    credentials: dict[str, Any] = {}
    if provider.value in config.providers:
        provider_config = config.providers[provider.value]

        if provider == LLMProvider.VERTEXAI:
            vertex_creds = provider_config.get("credentials")
            vertex_project = provider_config.get("project")
            vertex_location = provider_config.get("location")

            setup_vertex_environment(
                credentials=vertex_creds,
                project=vertex_project,
                location=vertex_location,
            )
        else:
            credentials = provider_config

    return credentials


async def _log_usage(
    db: Session,
    api_key_obj: APIKey | None,
    model: str,
    provider: str | None,
    endpoint: str,
    user_id: str | None = None,
    response: ChatCompletion | AsyncIterator[ChatCompletionChunk] | None = None,
    usage_override: CompletionUsage | None = None,
    error: str | None = None,
) -> None:
    """Log API usage to database and update user spend.

    Args:
        db: Database session
        api_key_obj: API key object (None if using master key)
        model: Model name
        provider: Provider name
        endpoint: Endpoint path
        user_id: User identifier for tracking
        response: Response object (if successful)
        usage_override: Usage data for streaming requests
        error: Error message (if failed)

    """
    usage_log = UsageLog(
        id=str(uuid.uuid4()),
        api_key_id=api_key_obj.id if api_key_obj else None,
        user_id=user_id,
        timestamp=datetime.now(UTC).replace(tzinfo=None),
        model=model,
        provider=provider,
        endpoint=endpoint,
        status="success" if error is None else "error",
        error_message=error,
    )

    usage_data = usage_override
    if not usage_data and response and isinstance(response, ChatCompletion) and response.usage:
        usage_data = response.usage

    if usage_data:
        usage_log.prompt_tokens = usage_data.prompt_tokens
        usage_log.completion_tokens = usage_data.completion_tokens
        usage_log.total_tokens = usage_data.total_tokens

        model_key = f"{provider}:{model}" if provider else model
        pricing = db.query(ModelPricing).filter(ModelPricing.model_key == model_key).first()

        if pricing:
            cost = (usage_data.prompt_tokens / 1_000_000) * pricing.input_price_per_million + (
                usage_data.completion_tokens / 1_000_000
            ) * pricing.output_price_per_million
            usage_log.cost = cost

            if user_id:
                user = db.query(User).filter(User.user_id == user_id).first()
                if user:
                    user.spend = float(user.spend) + cost
        else:
            logger.warning(f"No pricing configured for model '{model_key}'. Usage will be tracked without cost.")

    db.add(usage_log)
    try:
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log usage to database: {e}")
        db.rollback()


@router.post("/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ChatCompletion | StreamingResponse:
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming and non-streaming responses.
    Handles reasoning content from any-llm providers.

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

    _ = await validate_user_budget(db, user_id)

    provider, model = AnyLLM.split_model_provider(request.model)

    credentials = _get_provider_credentials(config, provider)

    completion_kwargs = request.model_dump()
    completion_kwargs.update(credentials)

    try:
        if request.stream:

            async def generate() -> AsyncIterator[str]:
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

                try:
                    stream: AsyncIterator[ChatCompletionChunk] = await acompletion(**completion_kwargs)  # type: ignore[assignment]
                    async for chunk in stream:
                        if chunk.usage:
                            # Prompt tokens should be constant, take first non-zero value
                            if chunk.usage.prompt_tokens and not prompt_tokens:
                                prompt_tokens = chunk.usage.prompt_tokens
                            if chunk.usage.completion_tokens:
                                completion_tokens = max(completion_tokens, chunk.usage.completion_tokens)
                            if chunk.usage.total_tokens:
                                total_tokens = max(total_tokens, chunk.usage.total_tokens)

                        yield f"data: {chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"

                    # Log aggregated usage
                    if prompt_tokens or completion_tokens or total_tokens:
                        usage_data = CompletionUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                        )
                        await _log_usage(
                            db=db,
                            api_key_obj=api_key,
                            model=model,
                            provider=provider,
                            endpoint="/v1/chat/completions",
                            user_id=user_id,
                            usage_override=usage_data,
                        )
                    else:
                        # This should never happen.
                        logger.warning(f"No usage data received from streaming response for model {model}")
                except Exception as e:
                    await _log_usage(
                        db=db,
                        api_key_obj=api_key,
                        model=model,
                        provider=provider,
                        endpoint="/v1/chat/completions",
                        user_id=user_id,
                        error=str(e),
                    )
                    raise

            return StreamingResponse(generate(), media_type="text/event-stream")

        response: ChatCompletion = await acompletion(**completion_kwargs)  # type: ignore[assignment]
        await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/chat/completions",
            user_id=user_id,
            response=response,
        )

    except Exception as e:
        await _log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/chat/completions",
            user_id=user_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calling provider: {e!s}",
        ) from e
    return response
