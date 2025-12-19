from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

import httpx  # noqa: TC002
from any_llm_platform_client import (
    AnyLLMPlatformClient,  # noqa: TC002
)

if TYPE_CHECKING:
    from any_llm.types.completion import ChatCompletion


ANY_LLM_PLATFORM_URL = os.getenv("ANY_LLM_PLATFORM_URL", "https://platform-api.any-llm.ai")
API_V1_STR = "/api/v1"
ANY_LLM_PLATFORM_API_URL = f"{ANY_LLM_PLATFORM_URL}{API_V1_STR}"


async def post_completion_usage_event(
    platform_client: AnyLLMPlatformClient,
    client: httpx.AsyncClient,
    any_llm_key: str,
    provider: str,
    completion: ChatCompletion,
    provider_key_id: str,
    client_name: str | None = None,
) -> None:
    """Posts completion usage events.

    The client uses the convenience method to get a solved challenge and prove ownership
    of the project it wants to post data to.

    Args:
        platform_client: The AnyLLMPlatformClient instance to use for authentication.
        client: An httpx client to perform post request.
        any_llm_key: The Any LLM platform key, tied to a specific project.
        provider: The name of the LLM provider.
        completion: The LLM response.
        provider_key_id: The unique identifier for the provider key.
        client_name: Optional name of the client for per-client usage tracking.
    """
    # Get solved challenge using the convenience method
    solved_challenge = await platform_client.aget_solved_challenge(any_llm_key=any_llm_key)

    # Get the public key for the request headers
    public_key = platform_client.get_public_key(any_llm_key)

    # Prepare and send usage event data
    payload = {
        "provider_key_id": provider_key_id,
        "provider": provider,
        "model": completion.model,
        "data": {
            "input_tokens": str(completion.usage.prompt_tokens),  # type: ignore[union-attr]
            "output_tokens": str(completion.usage.completion_tokens),  # type: ignore[union-attr]
        },
        "id": str(uuid.uuid4()),
    }
    if client_name:
        payload["client_name"] = client_name

    response = await client.post(
        f"{ANY_LLM_PLATFORM_API_URL}/usage-events/",
        json=payload,
        headers={"encryption-key": public_key, "AnyLLM-Challenge-Response": str(solved_challenge)},
    )
    response.raise_for_status()
