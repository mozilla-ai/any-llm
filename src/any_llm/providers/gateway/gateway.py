from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from any_llm.exceptions import (
    AuthenticationError,
    GatewayTimeoutError,
    InsufficientFundsError,
    ModelNotFoundError,
    RateLimitError,
    UpstreamProviderError,
)
from any_llm.providers.otari.otari import (
    GATEWAY_PLATFORM_TOKEN_ENV,
    LEGACY_GATEWAY_API_BASE_ENV,
    LEGACY_GATEWAY_API_KEY_ENV,
    LEGACY_GATEWAY_HEADER_NAME,
    OTARI_HEADER_NAME,
    OTARI_PLATFORM_TOKEN_ENV,
    OtariProvider,
    _extract_model_from_requests,
    _parse_jsonl_to_requests,
)

if TYPE_CHECKING:
    from any_llm.types.moderation import ModerationResponse

GATEWAY_HEADER_NAME = LEGACY_GATEWAY_HEADER_NAME

_STATUS_TO_EXCEPTION: dict[int, type[AuthenticationError | ModelNotFoundError]] = {
    401: AuthenticationError,
    403: AuthenticationError,
    404: ModelNotFoundError,
}


class GatewayProvider(OtariProvider):
    PROVIDER_NAME = "gateway"
    ENV_API_KEY_NAME = LEGACY_GATEWAY_API_KEY_ENV
    ENV_API_BASE_NAME = LEGACY_GATEWAY_API_BASE_ENV

    @override
    @classmethod
    def _resolve_env_api_base(cls) -> str | None:
        return os.getenv(LEGACY_GATEWAY_API_BASE_ENV) or os.getenv(OtariProvider.ENV_API_BASE_NAME)

    @override
    @classmethod
    def _resolve_env_api_key(cls) -> str | None:
        return os.getenv(LEGACY_GATEWAY_API_KEY_ENV) or os.getenv(OtariProvider.ENV_API_KEY_NAME)

    @override
    @staticmethod
    def _resolve_platform_token() -> str | None:
        return os.getenv(GATEWAY_PLATFORM_TOKEN_ENV) or os.getenv(OTARI_PLATFORM_TOKEN_ENV)

    def _handle_platform_error(self, exc: Exception) -> None:
        import openai

        if not isinstance(exc, openai.APIStatusError):
            raise exc

        status = exc.status_code
        headers = exc.response.headers
        correlation_id = headers.get("x-correlation-id")
        retry_after = headers.get("retry-after")

        detail = str(exc.message) if hasattr(exc, "message") else str(exc)
        if correlation_id:
            detail = f"{detail} (correlation_id={correlation_id})"

        if (exc_cls := _STATUS_TO_EXCEPTION.get(status)) is not None:
            raise exc_cls(
                message=detail,
                original_exception=exc,
                provider_name=self.PROVIDER_NAME,
            ) from exc

        if status == 402:
            raise InsufficientFundsError(
                message=detail,
                original_exception=exc,
                provider_name=self.PROVIDER_NAME,
            ) from exc

        if status == 429:
            raise RateLimitError(
                message=detail,
                original_exception=exc,
                provider_name=self.PROVIDER_NAME,
                retry_after=retry_after,
            ) from exc

        if status == 502:
            raise UpstreamProviderError(
                message=detail,
                original_exception=exc,
                provider_name=self.PROVIDER_NAME,
            ) from exc

        if status == 504:
            raise GatewayTimeoutError(
                message=detail,
                original_exception=exc,
                provider_name=self.PROVIDER_NAME,
            ) from exc

        raise exc

    @override
    async def _amoderation(
        self,
        model: str,
        input: str | list[str] | list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModerationResponse:
        if not self.platform_mode:
            return await super()._amoderation(model, input, **kwargs)
        try:
            return await super()._amoderation(model, input, **kwargs)
        except Exception as exc:
            self._handle_platform_error(exc)
            raise

    @override
    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        *,
        platform_mode: bool | None = None,
        **kwargs: Any,
    ) -> None:
        warnings.warn(
            "The 'gateway' provider is deprecated and will be removed in a future release. Use 'otari' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            platform_mode=platform_mode,
            **kwargs,
        )


__all__ = [
    "GATEWAY_HEADER_NAME",
    "GATEWAY_PLATFORM_TOKEN_ENV",
    "LEGACY_GATEWAY_API_BASE_ENV",
    "LEGACY_GATEWAY_API_KEY_ENV",
    "LEGACY_GATEWAY_HEADER_NAME",
    "OTARI_HEADER_NAME",
    "OTARI_PLATFORM_TOKEN_ENV",
    "GatewayProvider",
    "_extract_model_from_requests",
    "_parse_jsonl_to_requests",
]
