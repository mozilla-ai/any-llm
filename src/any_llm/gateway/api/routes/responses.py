from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from openai.types.responses import ResponseUsage
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from any_llm import AnyLLM, aresponses
from any_llm.gateway.api.deps import get_config, get_db, verify_api_key_or_master_key
from any_llm.gateway.api.routes._helpers import resolve_user_id
from any_llm.gateway.api.routes.chat import get_provider_kwargs, log_usage, rate_limit_headers
from any_llm.gateway.core.config import GatewayConfig
from any_llm.gateway.log_config import logger
from any_llm.gateway.models.entities import APIKey
from any_llm.gateway.rate_limit import check_rate_limit
from any_llm.gateway.services.budget_service import validate_user_budget
from any_llm.gateway.streaming import RESPONSES_STREAM_FORMAT, streaming_generator
from any_llm.types.completion import CompletionUsage
from any_llm.types.responses import ResponseStreamEvent

router = APIRouter(prefix="/v1", tags=["responses"])


class ResponsesRequest(BaseModel):
    """OpenAI-compatible responses request.

    ``extra="allow"`` lets Responses-only fields (reasoning, include, etc.)
    pass through without being declared explicitly on this model.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    input: Any
    stream: bool = False
    user: str | None = None


def _usage_to_completion_usage(usage: ResponseUsage | None) -> CompletionUsage | None:
    if usage is None:
        return None

    return CompletionUsage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )


@router.post("/responses", response_model=None)
async def create_response(
    raw_request: Request,
    response: Response,
    request: ResponsesRequest,
    auth_result: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[Session, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> dict[str, Any] | StreamingResponse:
    """OpenAI-compatible responses endpoint."""
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
    if not AnyLLM.get_provider_class(provider).SUPPORTS_RESPONSES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Provider '{provider.value}' does not support the Responses API",
        )

    provider_kwargs = get_provider_kwargs(config, provider)

    # Request fields take precedence over provider config defaults.
    request_fields = request.model_dump(exclude_unset=True)
    # OpenAI uses "input"; the SDK parameter is "input_data".
    request_fields["input_data"] = request_fields.pop("input")
    call_kwargs: dict[str, Any] = {**provider_kwargs, **request_fields}

    try:
        if request.stream:

            def _format_chunk(event: ResponseStreamEvent) -> str:
                return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"

            def _extract_usage(event: ResponseStreamEvent) -> CompletionUsage | None:
                if event.type != "response.completed":
                    return None
                response_obj = getattr(event, "response", None)
                return _usage_to_completion_usage(response_obj.usage if response_obj else None)

            async def _on_complete(usage_data: CompletionUsage) -> None:
                await log_usage(
                    db=db,
                    api_key_obj=api_key,
                    model=model,
                    provider=provider,
                    endpoint="/v1/responses",
                    user_id=user_id,
                    usage_override=usage_data,
                )

            async def _on_error(error: str) -> None:
                await log_usage(
                    db=db,
                    api_key_obj=api_key,
                    model=model,
                    provider=provider,
                    endpoint="/v1/responses",
                    user_id=user_id,
                    error=error,
                )

            response_stream = await aresponses(**call_kwargs)
            rl_headers = rate_limit_headers(rate_limit_info) if rate_limit_info else {}
            return StreamingResponse(
                streaming_generator(
                    stream=response_stream,
                    format_chunk=_format_chunk,
                    extract_usage=_extract_usage,
                    fmt=RESPONSES_STREAM_FORMAT,
                    on_complete=_on_complete,
                    on_error=_on_error,
                    label=f"{provider}:{model}",
                ),
                media_type="text/event-stream",
                headers=rl_headers,
            )

        result = await aresponses(**call_kwargs)
        usage_data = _usage_to_completion_usage(cast("ResponseUsage | None", result.usage))  # type: ignore[union-attr]
        await log_usage(
            db=db,
            api_key_obj=api_key,
            model=model,
            provider=provider,
            endpoint="/v1/responses",
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
            endpoint="/v1/responses",
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

    return cast("dict[str, Any]", result.model_dump(exclude_none=True))
