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
        from otari import errors as otari_errors

        # otari 0.1.0 raises its own typed exceptions (no longer openai.APIStatusError);
        # translate them into any-llm's equivalents, preserving the original as the cause.
        if not isinstance(exc, otari_errors.OtariError):
            raise exc

        common: dict[str, Any] = {
            "message": exc.message,
            "original_exception": exc,
            "provider_name": self.PROVIDER_NAME,
        }

        if isinstance(exc, otari_errors.AuthenticationError):
            raise AuthenticationError(**common) from exc
        if isinstance(exc, otari_errors.ModelNotFoundError):
            raise ModelNotFoundError(**common) from exc
        if isinstance(exc, otari_errors.InsufficientFundsError):
            raise InsufficientFundsError(**common) from exc
        if isinstance(exc, otari_errors.RateLimitError):
            raise RateLimitError(**common, retry_after=exc.retry_after) from exc
        if isinstance(exc, otari_errors.UpstreamProviderError):
            raise UpstreamProviderError(**common) from exc
        if isinstance(exc, otari_errors.GatewayTimeoutError):
            raise GatewayTimeoutError(**common) from exc

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
