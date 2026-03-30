import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from any_llm import AnyLLM, LLMProvider, acompletion
from any_llm.gateway.auth import verify_api_key_or_master_key
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.auth.vertex_auth import setup_vertex_environment
from any_llm.gateway.budget import validate_user_budget
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, UsageLog, User, get_db
from any_llm.gateway.log_config import logger
from any_llm.gateway.metrics import record_cost, record_tokens
from any_llm.gateway.pricing import find_model_pricing
from any_llm.gateway.rate_limit import RateLimitInfo, check_rate_limit
from any_llm.gateway.routes._helpers import resolve_user_id
from any_llm.gateway.streaming import OPENAI_STREAM_FORMAT, streaming_generator
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionUsage

router = APIRouter(prefix="/v1/chat", tags=["chat"])


def rate_limit_headers(info: RateLimitInfo) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(info.limit),
        "X-RateLimit-Remaining": str(info.remaining),
        "X-RateLimit-Reset": str(int(info.reset)),
    }


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)

    @field_validator("messages")
    @classmethod
    def validate_message_structure(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        for i, message in enumerate(v):
            if "role" not in message:
                msg = f"messages[{i}]: 'role' is required"
                raise ValueError(msg)
        return v

    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None


def get_provider_kwargs(
    config: GatewayConfig,
    provider: LLMProvider,
) -> dict[str, Any]:
    """Get provider kwargs from config for acompletion calls.

    Args:
        config: Gateway configuration
        provider: Provider name

    Returns:
        Dictionary of provider kwargs (credentials, client_args, etc.)

    """
    kwargs: dict[str, Any] = {}
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
            if "client_args" in provider_config:
                kwargs["client_args"] = provider_config["client_args"]
        else:
            kwargs = {k: v for k, v in provider_config.items() if k != "client_args"}
            if "client_args" in provider_config:
                kwargs["client_args"] = provider_config["client_args"]

    return kwargs


async def log_usage(
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
        timestamp=datetime.now(UTC),
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

        record_tokens(str(provider or ""), model, usage_data.prompt_tokens, usage_data.completion_tokens)

        pricing = find_model_pricing(db, provider, model)
        if pricing:
            cost = (usage_data.prompt_tokens / 1_000_000) * pricing.input_price_per_million + (
                usage_data.completion_tokens / 1_000_000
            ) * pricing.output_price_per_million
            usage_log.cost = cost
            record_cost(str(provider or ""), model, cost)

            if user_id:
                db.query(User).filter(User.user_id == user_id, User.deleted_at.is_(None)).update(
                    {User.spend: User.spend + cost}
                )
        else:
            model_ref = f"{provider}:{model}" if provider else model
            logger.warning(f"No pricing configured for '{model_ref}'. Usage will be tracked without cost.")

    try:
        db.add(usage_log)
        db.commit()
    except SQLAlchemyError as e:
        logger.error("Failed to log usage to database: %s", e)
        db.rollback()


@router.post("/completions", response_model=None)
async def chat_completions(
    raw_request: Request,
    response: Response,
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

    user_id = resolve_user_id(
        user_id_from_request=request.user,
        api_key=api_key,
        is_master_key=is_master_key,
        master_key_error=HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="When using master key, 'user' field is required in request body",
        ),
        no_api_key_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key validation failed",
        ),
        no_user_error=HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key has no associated user",
        ),
    )

    rate_limit_info = check_rate_limit(raw_request, user_id)

    _ = await validate_user_budget(db, user_id, request.model)

    provider, model = AnyLLM.split_model_provider(request.model)

    provider_kwargs = get_provider_kwargs(config, provider)

    # User request fields take precedence over provider config defaults
    request_fields = request.model_dump(exclude_unset=True)
    completion_kwargs = {**provider_kwargs, **request_fields}

    if request.stream and completion_kwargs.get("stream_options") is None:
        completion_kwargs["stream_options"] = {"include_usage": True}

    try:
        if request.stream:

            def _format_chunk(chunk: ChatCompletionChunk) -> str:
                return f"data: {chunk.model_dump_json()}\n\n"

            def _extract_usage(chunk: ChatCompletionChunk) -> CompletionUsage | None:
                if not chunk.usage:
                    return None
                return CompletionUsage(
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
                )

            async def _on_complete(usage_data: CompletionUsage) -> None:
                await log_usage(
                    db=db,
                    api_key_obj=api_key,
                    model=model,
                    provider=provider,
                    endpoint="/v1/chat/completions",
                    user_id=user_id,
                    usage_override=usage_data,
                )

            async def _on_error(error: str) -> None:
                await log_usage(
                    db=db,
                    api_key_obj=api_key,
                    model=model,
                    provider=provider,
                    endpoint="/v1/chat/completions",
                    user_id=user_id,
                    error=error,
                )

            stream: AsyncIterator[ChatCompletionChunk] = await acompletion(**completion_kwargs)  # type: ignore[assignment]
            rl_headers = rate_limit_headers(rate_limit_info) if rate_limit_info else {}
            return StreamingResponse(
                streaming_generator(
                    stream=stream,
                    format_chunk=_format_chunk,
                    extract_usage=_extract_usage,
                    fmt=OPENAI_STREAM_FORMAT,
                    on_complete=_on_complete,
                    on_error=_on_error,
                    label=f"{provider}:{model}",
                ),
                media_type="text/event-stream",
                headers=rl_headers,
            )

        completion: ChatCompletion = await acompletion(**completion_kwargs)  # type: ignore[assignment]
        await log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/chat/completions",
            user_id=user_id,
            response=completion,
        )

    except Exception as e:
        await log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/chat/completions",
            user_id=user_id,
            error=str(e),
        )
        logger.error(f"Provider call failed for {provider}:{model}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The request could not be completed by the provider",
        ) from e

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    return completion
