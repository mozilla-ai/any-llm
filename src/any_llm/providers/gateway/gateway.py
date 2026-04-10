from __future__ import annotations

import os
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
from any_llm.logging import logger
from any_llm.providers.openai.base import BaseOpenAIProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from openresponses_types import ResponseResource

    from any_llm.types.completion import (
        ChatCompletion,
        ChatCompletionChunk,
        CompletionParams,
        CreateEmbeddingResponse,
    )
    from any_llm.types.model import Model
    from any_llm.types.responses import Response, ResponsesParams, ResponseStreamEvent

GATEWAY_HEADER_NAME = "X-AnyLLM-Key"
GATEWAY_PLATFORM_TOKEN_ENV = "GATEWAY_PLATFORM_TOKEN"  # noqa: S105

_STATUS_TO_EXCEPTION: dict[int, type[AuthenticationError | ModelNotFoundError]] = {
    401: AuthenticationError,
    403: AuthenticationError,
    404: ModelNotFoundError,
}


class GatewayProvider(BaseOpenAIProvider):
    ENV_API_KEY_NAME = "GATEWAY_API_KEY"
    ENV_API_BASE_NAME = "GATEWAY_API_BASE"
    PROVIDER_NAME = "gateway"
    PROVIDER_DOCUMENTATION_URL = "https://github.com/mozilla-ai/any-llm"

    # All features are marked as supported, but depending on which provider
    # you call inside the gateway, they may not all work.
    SUPPORTS_COMPLETION_STREAMING = True
    SUPPORTS_COMPLETION = True
    SUPPORTS_RESPONSES = True
    SUPPORTS_COMPLETION_REASONING = True
    SUPPORTS_COMPLETION_IMAGE = True
    SUPPORTS_COMPLETION_PDF = True
    SUPPORTS_EMBEDDING = True
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = True

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        *,
        platform_mode: bool | None = None,
        **kwargs: Any,
    ) -> None:
        resolved_api_base = api_base or os.getenv(self.ENV_API_BASE_NAME)
        if not resolved_api_base:
            msg = f"For any-llm-gateway, api_base is required (set via parameter or {self.ENV_API_BASE_NAME} env var)"
            raise ValueError(msg)

        platform_token = os.getenv(GATEWAY_PLATFORM_TOKEN_ENV)

        if platform_mode is True:
            resolved_token = api_key or platform_token
            if not resolved_token:
                msg = f"Platform mode requires a user token (pass api_key or set the {GATEWAY_PLATFORM_TOKEN_ENV} env var)"
                raise ValueError(msg)
            self.platform_mode = True
            super().__init__(api_key=resolved_token, api_base=resolved_api_base, **kwargs)
            return

        if platform_mode is None and platform_token and not api_key:
            self.platform_mode = True
            super().__init__(api_key=platform_token, api_base=resolved_api_base, **kwargs)
            return

        # Non-platform mode (existing behavior)
        self.platform_mode = False
        api_key = self._verify_and_set_api_key(api_key)
        if api_key:
            if "default_headers" not in kwargs:
                kwargs["default_headers"] = {}
            elif kwargs["default_headers"].get(GATEWAY_HEADER_NAME):
                msg = f"{GATEWAY_HEADER_NAME} header is already set, overriding with new API key"
                logger.info(msg)
            kwargs["default_headers"][GATEWAY_HEADER_NAME] = f"Bearer {api_key}"
        super().__init__(api_key=api_key, api_base=resolved_api_base, **kwargs)

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        """Unlike other providers, the gateway provider does not require an API key."""
        return api_key or os.getenv(self.ENV_API_KEY_NAME, "")

    # -- Platform-mode error handling -----------------------------------------

    def _handle_platform_error(self, exc: Exception) -> None:
        """Convert ``openai.APIStatusError`` to typed any-llm exceptions.

        Extracts the ``Retry-After`` and ``X-Correlation-ID`` response headers
        when available and includes them in the raised exception so callers can
        act on rate-limit back-off or trace gateway requests.
        """
        import openai  # inline import to keep module-level deps minimal

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

    # -- Overridden async methods with platform error wrapping ----------------

    @override
    async def _acompletion(
        self, params: CompletionParams, **kwargs: Any
    ) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
        if not self.platform_mode:
            return await super()._acompletion(params, **kwargs)
        try:
            return await super()._acompletion(params, **kwargs)
        except Exception as exc:
            self._handle_platform_error(exc)
            raise  # unreachable when _handle_platform_error raises, satisfies type checker

    @override
    async def _aresponses(
        self, params: ResponsesParams, **kwargs: Any
    ) -> ResponseResource | Response | AsyncIterator[ResponseStreamEvent]:
        if not self.platform_mode:
            return await super()._aresponses(params, **kwargs)
        try:
            return await super()._aresponses(params, **kwargs)
        except Exception as exc:
            self._handle_platform_error(exc)
            raise

    @override
    async def _aembedding(
        self,
        model: str,
        inputs: str | list[str],
        **kwargs: Any,
    ) -> CreateEmbeddingResponse:
        if not self.platform_mode:
            return await super()._aembedding(model, inputs, **kwargs)
        try:
            return await super()._aembedding(model, inputs, **kwargs)
        except Exception as exc:
            self._handle_platform_error(exc)
            raise

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        if not self.platform_mode:
            return await super()._alist_models(**kwargs)
        try:
            return await super()._alist_models(**kwargs)
        except Exception as exc:
            self._handle_platform_error(exc)
            raise
