import json
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from any_llm import AnyLLM, amessages
from any_llm.gateway.auth import verify_api_key_or_master_key
from any_llm.gateway.auth.dependencies import get_config
from any_llm.gateway.budget import validate_user_budget
from any_llm.gateway.config import GatewayConfig
from any_llm.gateway.db import APIKey, get_db
from any_llm.gateway.log_config import logger
from any_llm.gateway.rate_limit import check_rate_limit
from any_llm.gateway.routes.chat import get_provider_kwargs, log_usage, rate_limit_headers
from any_llm.types.completion import CompletionUsage
from any_llm.types.messages import MessageResponse, MessageStreamEvent  # noqa: TC001

router = APIRouter(prefix="/v1", tags=["messages"])


class MessagesRequest(BaseModel):
    """Anthropic Messages API-compatible request."""

    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)
    max_tokens: int
    system: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    stop_sequences: list[str] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    thinking: dict[str, Any] | None = None


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


def _resolve_user_id(
    request: MessagesRequest,
    api_key: APIKey | None,
    is_master_key: bool,
) -> str:
    """Resolve user_id from request metadata, API key, or master key context."""
    user_from_metadata = request.metadata.get("user_id") if request.metadata else None

    if is_master_key:
        if not user_from_metadata:
            raise _anthropic_error(
                _ERR_INVALID_REQUEST,
                _MASTER_KEY_USER_REQUIRED,
                status.HTTP_400_BAD_REQUEST,
            )
        return str(user_from_metadata)

    if user_from_metadata:
        return str(user_from_metadata)

    if api_key is None:
        raise _anthropic_error(_ERR_API, _API_KEY_VALIDATION_FAILED, status.HTTP_500_INTERNAL_SERVER_ERROR)
    if not api_key.user_id:
        raise _anthropic_error(_ERR_API, _API_KEY_NO_USER, status.HTTP_500_INTERNAL_SERVER_ERROR)
    return str(api_key.user_id)


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
    user_id = _resolve_user_id(request, api_key, is_master_key)

    rate_limit_info = check_rate_limit(raw_request, user_id)

    _ = await validate_user_budget(db, user_id, request.model)

    provider, model = AnyLLM.split_model_provider(request.model)

    provider_kwargs = get_provider_kwargs(config, provider)

    call_kwargs: dict[str, Any] = {
        "model": f"{provider.value}:{model}",
        "messages": request.messages,
        "max_tokens": request.max_tokens,
    }
    if request.system is not None:
        call_kwargs["system"] = request.system
    if request.temperature is not None:
        call_kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        call_kwargs["top_p"] = request.top_p
    if request.top_k is not None:
        call_kwargs["top_k"] = request.top_k
    if request.stop_sequences is not None:
        call_kwargs["stop_sequences"] = request.stop_sequences
    if request.tools is not None:
        call_kwargs["tools"] = request.tools
    if request.tool_choice is not None:
        call_kwargs["tool_choice"] = request.tool_choice
    if request.metadata is not None:
        call_kwargs["metadata"] = request.metadata
    if request.thinking is not None:
        call_kwargs["thinking"] = request.thinking

    # Pass provider credentials
    if "api_key" in provider_kwargs:
        call_kwargs["api_key"] = provider_kwargs["api_key"]
    if "api_base" in provider_kwargs:
        call_kwargs["api_base"] = provider_kwargs["api_base"]
    client_args = provider_kwargs.get("client_args")
    if client_args:
        call_kwargs["client_args"] = client_args

    try:
        if request.stream:
            call_kwargs["stream"] = True

            async def generate() -> AsyncIterator[str]:
                input_tokens = 0
                output_tokens = 0

                try:
                    stream: AsyncIterator[MessageStreamEvent] = await amessages(**call_kwargs)  # type: ignore[assignment]
                    async for event in stream:
                        if event.usage:
                            if event.usage.input_tokens:
                                input_tokens = event.usage.input_tokens
                            if event.usage.output_tokens:
                                output_tokens = event.usage.output_tokens
                        yield f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"
                    yield "event: done\ndata: {}\n\n"

                    if input_tokens or output_tokens:
                        usage_data = CompletionUsage(
                            prompt_tokens=input_tokens,
                            completion_tokens=output_tokens,
                            total_tokens=input_tokens + output_tokens,
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
                except Exception as e:
                    error_data = {
                        "type": "error",
                        "error": {"type": "api_error", "message": "An error occurred during streaming"},
                    }
                    yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
                    try:
                        await log_usage(
                            db=db,
                            api_key_obj=api_key,
                            model=model,
                            provider=provider,
                            endpoint="/v1/messages",
                            user_id=user_id,
                            error=str(e),
                        )
                    except Exception as log_err:
                        logger.error(f"Failed to log streaming error usage: {log_err}")
                    logger.error(f"Streaming error for {provider}:{model}: {e}")

            rl_headers = rate_limit_headers(rate_limit_info) if rate_limit_info else {}
            return StreamingResponse(generate(), media_type="text/event-stream", headers=rl_headers)

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
