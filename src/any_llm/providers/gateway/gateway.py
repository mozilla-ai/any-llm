from __future__ import annotations

import warnings
from typing import Any

from typing_extensions import override

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

GATEWAY_HEADER_NAME = OTARI_HEADER_NAME


class GatewayProvider(OtariProvider):
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
