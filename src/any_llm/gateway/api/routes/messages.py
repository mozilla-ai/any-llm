from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from any_llm import AnyLLM, amessages
from any_llm.gateway.api.deps import verify_api_key_or_master_key
from any_llm.gateway.api.deps import get_config
from any_llm.gateway.services.budget_service import validate_user_budget
from any_llm.gateway.core.config import GatewayConfig
from any_llm.gateway.api.deps import get_db
from any_llm.gateway.models.entities import APIKey
from any_llm.gateway.log_config import logger
from any_llm.gateway.rate_limit import check_rate_limit
from any_llm.gateway.api.routes._helpers import resolve_user_id
from any_llm.gateway.api.routes.chat import get_provider_kwargs, log_usage, rate_limit_headers
from any_llm.gateway.streaming import ANTHROPIC_STREAM_FORMAT, streaming_generator
from any_llm.types.completion import CompletionUsage
from any_llm.types.messages import (
    MessageDeltaEvent,
    MessageResponse,
    MessageStartEvent,
    MessageStreamEvent,
)

router = APIRouter(prefix="/v1", tags=["messages"])


class MessagesRequest(BaseModel):
    """Anthropic Messages API-compatible request."""

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    max_tokens: int
    system: str | list[dict[str, Any]] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None
    cache_control: dict[str, Any] | None = None


def _anthropic_error(error_type: str, message: str, status_code: int) -> HTTPException:
    """Create an HTTPException with Anthropic-style error body."""
    return HTTPException(
        status_code=status_code,
        detail={"type": "error", "error": {"type": error_type, "message": message}},
    )


_ERR_INVALID_REQUEST = "invalid_request_error"
_ERR_API = "api_error"
_MASTER_KEY_USER_REQUIRED = "When using master key, 'metadata.user_id' is required in request body"
_API_KEY_VALIDATION_FAILED = "API key validation failed"
_API_KEY_NO_USER = "API key has no associated user"
_PROVIDER_ERROR = "The request could not be completed by the provider"


@router.post("/messages", response_model=None)
async def create_message(
    raw_request: Request,
    response: Response,
    request: MessagesRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> dict[str, Any] | StreamingResponse:
    """Anthropic Messages API-compatible endpoint."""
    api_key, is_master_key = auth_result
    user_from_metadata = request.metadata.get("user_id") if request.metadata else None
    user_id = resolve_user_id(
        user_id_from_request=str(user_from_metadata) if user_from_metadata else None,
        api_key=api_key,
        is_master_key=is_master_key,
        master_key_error=_anthropic_error(
            _ERR_INVALID_REQUEST,
            _MASTER_KEY_USER_REQUIRED,
            status.HTTP_400_BAD_REQUEST,
        ),
        no_api_key_error=_anthropic_error(
            _ERR_API,
            _API_KEY_VALIDATION_FAILED,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ),
        no_user_error=_anthropic_error(
            _ERR_API,
            _API_KEY_NO_USER,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ),
    )

    rate_limit_info = check_rate_limit(raw_request, user_id)

    _ = await validate_user_budget(db, user_id, request.model)

    provider, model = AnyLLM.split_model_provider(request.model)

    provider_kwargs = get_provider_kwargs(config, provider)

    # Request fields take precedence over provider config defaults
    request_fields = request.model_dump(exclude_unset=True)
    call_kwargs: dict[str, Any] = {**provider_kwargs, **request_fields}

    try:
        if request.stream:
            call_kwargs["stream"] = True

            def _format_chunk(event: MessageStreamEvent) -> str:
                return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"

            def _extract_usage(event: MessageStreamEvent) -> CompletionUsage | None:
                if isinstance(event, MessageDeltaEvent):
                    input_tokens = event.usage.input_tokens or 0
                    output_tokens = event.usage.output_tokens or 0
                    return CompletionUsage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    )
                if isinstance(event, MessageStartEvent):
                    input_tokens = event.message.usage.input_tokens or 0
                    if input_tokens:
                        return CompletionUsage(
                            prompt_tokens=input_tokens,
                            completion_tokens=0,
                            total_tokens=input_tokens,
                        )
                return None

            async def _on_complete(usage_data: CompletionUsage) -> None:
                await log_usage(
                    db=db,
                    api_key_obj=api_key,
                    model=model,
                    provider=provider,
                    endpoint="/v1/messages",
                    user_id=user_id,
                    usage_override=usage_data,
                )

            async def _on_error(error: str) -> None:
                await log_usage(
                    db=db,
                    api_key_obj=api_key,
                    model=model,
                    provider=provider,
                    endpoint="/v1/messages",
                    user_id=user_id,
                    error=error,
                )

            msg_stream = await amessages(**call_kwargs)
            rl_headers = rate_limit_headers(rate_limit_info) if rate_limit_info else {}
            return StreamingResponse(
                streaming_generator(
                    stream=msg_stream,  # type: ignore[arg-type]
                    format_chunk=_format_chunk,
                    extract_usage=_extract_usage,
                    fmt=ANTHROPIC_STREAM_FORMAT,
                    on_complete=_on_complete,
                    on_error=_on_error,
                    label=f"{provider}:{model}",
                ),
                media_type="text/event-stream",
                headers=rl_headers,
            )

        result: MessageResponse = await amessages(**call_kwargs)  # type: ignore[assignment]

        if result.usage:
            usage_data = CompletionUsage(
                prompt_tokens=result.usage.input_tokens,
                completion_tokens=result.usage.output_tokens,
                total_tokens=result.usage.input_tokens + result.usage.output_tokens,
            )
            await log_usage(
                db=db,
                api_key_obj=api_key,
                model=model,
                provider=provider,
                endpoint="/v1/messages",
                user_id=user_id,
                usage_override=usage_data,
            )

    except HTTPException:
        raise
    except Exception as e:
        await log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/messages",
            user_id=user_id,
            error=str(e),
        )
        logger.error(f"Provider call failed for {provider}:{model}: {e}")
        raise _anthropic_error(
            _ERR_API,
            _PROVIDER_ERROR,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        ) from e

    if rate_limit_info:
        for key, value in rate_limit_headers(rate_limit_info).items():
            response.headers[key] = value

    return result.model_dump(exclude_none=True)
