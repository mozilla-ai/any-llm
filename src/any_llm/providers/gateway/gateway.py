from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from any_llm.exceptions import (
    AuthenticationError,
    BatchNotCompleteError,
    GatewayTimeoutError,
    InsufficientFundsError,
    InvalidRequestError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    UpstreamProviderError,
)
from any_llm.logging import logger
from any_llm.providers.openai.base import BaseOpenAIProvider
from any_llm.types.batch import Batch, BatchResult, BatchResultError, BatchResultItem
from any_llm.types.completion import ChatCompletion

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from openresponses_types import ResponseResource

    from any_llm.types.completion import (
        ChatCompletionChunk,
        CompletionParams,
        CreateEmbeddingResponse,
    )
    from any_llm.types.model import Model
    from any_llm.types.moderation import ModerationResponse
    from any_llm.types.responses import Response, ResponsesParams, ResponseStreamEvent

GATEWAY_HEADER_NAME = "X-AnyLLM-Key"
GATEWAY_PLATFORM_TOKEN_ENV = "GATEWAY_PLATFORM_TOKEN"  # noqa: S105

_STATUS_TO_EXCEPTION: dict[int, type[AuthenticationError | ModelNotFoundError]] = {
    401: AuthenticationError,
    403: AuthenticationError,
    404: ModelNotFoundError,
}


def _parse_jsonl_to_requests(file_path: str) -> list[dict[str, Any]]:
    """Parse a JSONL file into a list of batch request objects."""
    requests: list[dict[str, Any]] = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            requests.append(
                {
                    "custom_id": entry["custom_id"],
                    "body": entry.get("body", {}),
                }
            )
    return requests


def _extract_model_from_requests(requests: list[dict[str, Any]]) -> str | None:
    """Extract the model from the first request's body."""
    if requests and requests[0].get("body"):
        model: Any = requests[0]["body"].get("model")
        return str(model) if model is not None else None
    return None


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
    SUPPORTS_MODERATION = True
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
    async def _amoderation(
        self,
        model: str,
        input: str | list[str] | list[dict[str, Any]],  # noqa: A002
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
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        if not self.platform_mode:
            return await super()._alist_models(**kwargs)
        try:
            return await super()._alist_models(**kwargs)
        except Exception as exc:
            self._handle_platform_error(exc)
            raise

    # -- Batch API overrides --------------------------------------------------

    def _handle_batch_http_error(self, response: Any) -> None:
        """Handle HTTP errors from gateway batch endpoints."""
        if response.status_code == 404:
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except Exception:  # noqa: S110
                pass
            if "batches" in str(response.url):
                msg = (
                    "This gateway does not support batch operations. "
                    "Upgrade your gateway to a version that supports /v1/batches endpoints."
                )
                raise ProviderError(message=msg, provider_name=self.PROVIDER_NAME)
            raise ProviderError(message=detail or "Not found", provider_name=self.PROVIDER_NAME)
        response.raise_for_status()

    @override
    async def _acreate_batch(
        self,
        input_file_path: str,
        endpoint: str,
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Batch:
        """Create a batch job via the gateway's JSON batch endpoint."""
        requests = _parse_jsonl_to_requests(input_file_path)
        model = kwargs.pop("model", None)
        if not model:
            model = _extract_model_from_requests(requests)

        body: dict[str, Any] = {
            "model": model,
            "requests": requests,
            "completion_window": completion_window,
        }
        if metadata:
            body["metadata"] = metadata

        response = await self.client._client.post(
            f"{self.client.base_url}batches",
            json=body,
        )
        if response.status_code == 404:
            msg = (
                "This gateway does not support batch operations. "
                "Upgrade your gateway to a version that supports /v1/batches endpoints."
            )
            raise ProviderError(message=msg, provider_name=self.PROVIDER_NAME)
        response.raise_for_status()
        data = response.json()
        return Batch(**{k: v for k, v in data.items() if k != "provider"})

    @override
    async def _aretrieve_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Retrieve a batch job via the gateway."""
        provider_name = kwargs.pop("provider_name", None)
        if not provider_name:
            msg = "provider_name is required for Gateway batch operations"
            raise InvalidRequestError(message=msg, provider_name=self.PROVIDER_NAME)
        response = await self.client._client.get(
            f"{self.client.base_url}batches/{batch_id}",
            params={"provider": provider_name},
        )
        self._handle_batch_http_error(response)
        return Batch(**response.json())

    @override
    async def _acancel_batch(self, batch_id: str, **kwargs: Any) -> Batch:
        """Cancel a batch job via the gateway."""
        provider_name = kwargs.pop("provider_name", None)
        if not provider_name:
            msg = "provider_name is required for Gateway batch operations"
            raise InvalidRequestError(message=msg, provider_name=self.PROVIDER_NAME)
        response = await self.client._client.post(
            f"{self.client.base_url}batches/{batch_id}/cancel",
            params={"provider": provider_name},
        )
        self._handle_batch_http_error(response)
        return Batch(**response.json())

    @override
    async def _alist_batches(
        self,
        after: str | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> Sequence[Batch]:
        """List batch jobs via the gateway."""
        provider_name = kwargs.pop("provider_name", None)
        if not provider_name:
            msg = "provider_name is required for Gateway batch operations"
            raise InvalidRequestError(message=msg, provider_name=self.PROVIDER_NAME)
        params: dict[str, Any] = {"provider": provider_name}
        if after:
            params["after"] = after
        if limit is not None:
            params["limit"] = limit
        response = await self.client._client.get(
            f"{self.client.base_url}batches",
            params=params,
        )
        self._handle_batch_http_error(response)
        data = response.json()
        return [Batch(**b) for b in data.get("data", [])]

    @override
    async def _aretrieve_batch_results(self, batch_id: str, **kwargs: Any) -> BatchResult:
        """Retrieve the results of a completed batch job via the gateway."""
        provider_name = kwargs.pop("provider_name", None)
        if not provider_name:
            msg = "provider_name is required for Gateway batch operations"
            raise InvalidRequestError(message=msg, provider_name=self.PROVIDER_NAME)
        response = await self.client._client.get(
            f"{self.client.base_url}batches/{batch_id}/results",
            params={"provider": provider_name},
        )
        if response.status_code == 409:
            raise BatchNotCompleteError(
                batch_id=batch_id,
                status="unknown",
                provider_name=self.PROVIDER_NAME,
            )
        self._handle_batch_http_error(response)
        data = response.json()
        return BatchResult(
            results=[
                BatchResultItem(
                    custom_id=item["custom_id"],
                    result=ChatCompletion(**item["result"]) if item.get("result") else None,
                    error=BatchResultError(**item["error"]) if item.get("error") else None,
                )
                for item in data.get("results", [])
            ]
        )
