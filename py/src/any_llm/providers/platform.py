"""Platform Provider - Transparent proxy layer for centralized key management.

The Platform Provider wraps any existing provider to add centralized key management
and usage tracking. Users authenticate with a single platform key (ANY_LLM_KEY),
which the platform exchanges for provider-specific credentials at runtime.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

import httpx
from nacl.bindings import (
    crypto_aead_xchacha20poly1305_ietf_decrypt,
    crypto_scalarmult,
    crypto_scalarmult_base,
)

from any_llm.any_llm import AnyLLM
from any_llm.errors import AnyLLMError, AuthenticationError
from any_llm.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    CompletionParams,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

logger = logging.getLogger(__name__)

PROVIDER_NAME = "Platform"
ENV_VAR_NAME = "ANY_LLM_KEY"
DEFAULT_PLATFORM_BASE_URL = "http://localhost:8000/api/v1"

# Regex pattern to validate ANY_LLM_KEY format
ANY_LLM_KEY_PATTERN = re.compile(r"^ANY\.v1\.([^.]+)\.([^-]+)-(.+)$")


class PlatformError(AnyLLMError):
    """Raised when a platform-specific error occurs."""

    default_message = "Platform error"


class InvalidKeyFormatError(PlatformError):
    """Raised when the ANY_LLM_KEY format is invalid."""

    default_message = "Invalid ANY_LLM_KEY format"


class DecryptionError(PlatformError):
    """Raised when sealed box decryption fails."""

    default_message = "Failed to decrypt sealed box"


class ChallengeError(PlatformError):
    """Raised when the authentication challenge fails."""

    default_message = "Authentication challenge failed"


class TokenExpiredError(PlatformError):
    """Raised when the access token has expired."""

    default_message = "Access token expired"


class ProviderKeyNotFoundError(PlatformError):
    """Raised when the requested provider key is not found."""

    default_message = "Provider key not found"


@dataclass
class KeyComponents:
    """Parsed components from an ANY_LLM_KEY."""

    key_id: str
    public_key_fingerprint: str
    base64_encoded_private_key: str


@dataclass
class DecryptedProviderKey:
    """Decrypted provider API key with metadata."""

    api_key: str
    provider_key_id: UUID
    project_id: UUID
    provider: str
    created_at: datetime
    updated_at: datetime | None


def parse_any_llm_key(any_llm_key: str) -> KeyComponents:
    """Parse ANY_LLM_KEY into its components.

    Args:
        any_llm_key: The full ANY_LLM_KEY string in format
            ANY.v1.<kid>.<fingerprint>-<base64_key>

    Returns:
        KeyComponents with key_id, public_key_fingerprint, and base64_encoded_private_key

    Raises:
        InvalidKeyFormatError: If the key format is invalid

    """
    match = ANY_LLM_KEY_PATTERN.match(any_llm_key)
    if not match:
        raise InvalidKeyFormatError
    key_id, fingerprint, private_key_b64 = match.groups()
    return KeyComponents(
        key_id=key_id,
        public_key_fingerprint=fingerprint,
        base64_encoded_private_key=private_key_b64,
    )


def load_private_key(private_key_base64: str) -> bytes:
    """Load X25519 private key from base64.

    Args:
        private_key_base64: Base64-encoded 32-byte private key

    Returns:
        32-byte private key as bytes

    Raises:
        InvalidKeyFormatError: If the key is not exactly 32 bytes

    """
    try:
        private_key_bytes = base64.b64decode(private_key_base64)
    except Exception as e:
        msg = f"Failed to decode private key: {e}"
        raise InvalidKeyFormatError(message=msg) from e

    if len(private_key_bytes) != 32:
        msg = f"X25519 private key must be 32 bytes, got {len(private_key_bytes)}"
        raise InvalidKeyFormatError(message=msg)
    return private_key_bytes


def extract_public_key(private_key: bytes) -> str:
    """Derive public key from X25519 private key.

    Uses Curve25519 scalar multiplication: public = base_point * private

    Args:
        private_key: 32-byte X25519 private key

    Returns:
        Base64-encoded 32-byte public key

    """
    public_key_bytes = crypto_scalarmult_base(private_key)
    return base64.b64encode(public_key_bytes).decode("utf-8")


def decrypt_sealed_box(encrypted_data_base64: str, private_key: bytes) -> str:
    """Decrypt X25519 sealed box data.

    Sealed boxes use an ephemeral sender key for anonymous encryption.
    Format: [ephemeral_public_key(32)][ciphertext(N+16)]

    Args:
        encrypted_data_base64: Base64-encoded sealed box
        private_key: 32-byte X25519 private key

    Returns:
        Decrypted UTF-8 string

    Raises:
        DecryptionError: If format is invalid or decryption fails

    """
    try:
        encrypted_data = base64.b64decode(encrypted_data_base64)
    except Exception as e:
        msg = f"Failed to decode encrypted data: {e}"
        raise DecryptionError(message=msg) from e

    if len(encrypted_data) < 32:
        raise DecryptionError

    # Split sealed box components
    ephemeral_public_key = encrypted_data[:32]
    ciphertext = encrypted_data[32:]

    # Derive recipient public key from private key
    recipient_public_key = crypto_scalarmult_base(private_key)

    # Compute shared secret: shared = ephemeral_pk * private_key
    shared_secret = crypto_scalarmult(private_key, ephemeral_public_key)

    # Generate nonce: SHA-512(ephemeral_pk || recipient_pk)[:24]
    combined = ephemeral_public_key + recipient_public_key
    nonce = hashlib.sha512(combined).digest()[:24]

    try:
        # Decrypt with XChaCha20-Poly1305
        plaintext = crypto_aead_xchacha20poly1305_ietf_decrypt(
            ciphertext, None, nonce, shared_secret
        )
        return plaintext.decode("utf-8")
    except Exception as e:
        msg = f"Decryption failed: {e}"
        raise DecryptionError(message=msg) from e


def _parse_iso8601(value: str | None) -> datetime | None:
    """Parse ISO8601 timestamp string to datetime.

    Args:
        value: ISO8601 timestamp string or None

    Returns:
        Parsed datetime with UTC timezone or None

    """
    if not value:
        return None
    # Handle both 'Z' and '+00:00' formats
    value = value.replace("Z", "+00:00")
    return datetime.fromisoformat(value)


class PlatformClient:
    """Client for interacting with the ANY LLM Platform API.

    Handles authentication, token management, and provider key retrieval.
    """

    def __init__(self, base_url: str = DEFAULT_PLATFORM_BASE_URL) -> None:
        """Initialize the platform client.

        Args:
            base_url: Base URL for the platform API

        """
        self.base_url = base_url.rstrip("/")
        self._access_token: str | None = None
        self._token_expires_at: datetime | None = None
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def _create_challenge(self, public_key: str) -> str:
        """Create authentication challenge.

        Args:
            public_key: Base64-encoded public key

        Returns:
            Encrypted challenge (base64-encoded sealed box containing UUID)

        Raises:
            ChallengeError: If challenge creation fails

        """
        client = await self._get_client()
        url = f"{self.base_url}/auth/"

        try:
            response = await client.post(url, json={"encryption_key": public_key})
            if response.status_code == 400:
                msg = "Invalid public key format"
                raise ChallengeError(message=msg)
            if response.status_code == 404:
                msg = "No project found for the provided public key"
                raise ChallengeError(message=msg)
            response.raise_for_status()
            data = response.json()
            encrypted_challenge: str = data["encrypted_challenge"]
            return encrypted_challenge
        except httpx.HTTPStatusError as e:
            msg = f"Challenge creation failed: {e}"
            raise ChallengeError(message=msg) from e
        except httpx.RequestError as e:
            msg = f"Network error during challenge creation: {e}"
            raise ChallengeError(message=msg) from e

    async def _request_token(self, solved_challenge: str) -> tuple[str, int]:
        """Request access token with solved challenge.

        Args:
            solved_challenge: The decrypted UUID from the challenge

        Returns:
            Tuple of (access_token, expires_in_seconds)

        Raises:
            ChallengeError: If token request fails

        """
        client = await self._get_client()
        url = f"{self.base_url}/auth/token"

        try:
            response = await client.post(url, json={"solved_challenge": solved_challenge})
            if response.status_code == 400:
                msg = "Invalid challenge format"
                raise ChallengeError(message=msg)
            if response.status_code == 401:
                msg = "Challenge not found or expired"
                raise ChallengeError(message=msg)
            response.raise_for_status()
            data = response.json()
            return data["access_token"], data.get("expires_in", 86400)
        except httpx.HTTPStatusError as e:
            msg = f"Token request failed: {e}"
            raise ChallengeError(message=msg) from e
        except httpx.RequestError as e:
            msg = f"Network error during token request: {e}"
            raise ChallengeError(message=msg) from e

    async def authenticate(self, any_llm_key: str) -> str:
        """Perform full authentication flow.

        Args:
            any_llm_key: The ANY_LLM_KEY string

        Returns:
            Access token

        """
        # Parse key components
        components = parse_any_llm_key(any_llm_key)
        private_key = load_private_key(components.base64_encoded_private_key)
        public_key = extract_public_key(private_key)

        # Create challenge
        encrypted_challenge = await self._create_challenge(public_key)

        # Solve challenge by decrypting
        solved_challenge = decrypt_sealed_box(encrypted_challenge, private_key)

        # Request token
        access_token, expires_in = await self._request_token(solved_challenge)

        # Cache token with safety margin (23 hours instead of 24)
        self._access_token = access_token
        safety_margin_seconds = min(3600, expires_in // 24)  # 1 hour or 1/24th of expiry
        self._token_expires_at = datetime.now(tz=UTC) + timedelta(
            seconds=expires_in - safety_margin_seconds
        )

        return access_token

    async def ensure_valid_token(self, any_llm_key: str) -> str:
        """Get valid access token, refreshing if needed.

        Args:
            any_llm_key: The ANY_LLM_KEY string

        Returns:
            Valid access token string

        """
        now = datetime.now(tz=UTC)

        if (
            self._access_token is None
            or self._token_expires_at is None
            or now >= self._token_expires_at
        ):
            await self.authenticate(any_llm_key)

        return self._access_token  # type: ignore[return-value]

    async def get_provider_key(
        self, any_llm_key: str, provider: str
    ) -> DecryptedProviderKey:
        """Get decrypted provider key.

        Args:
            any_llm_key: The ANY_LLM_KEY string
            provider: Provider name (e.g., "openai", "anthropic")

        Returns:
            DecryptedProviderKey with api_key and metadata

        Raises:
            ProviderKeyNotFoundError: If the provider key is not found
            AuthenticationError: If authentication fails

        """
        access_token = await self.ensure_valid_token(any_llm_key)
        client = await self._get_client()
        url = f"{self.base_url}/provider-keys/{provider}"

        try:
            response = await client.get(
                url, headers={"Authorization": f"Bearer {access_token}"}
            )
            if response.status_code == 401:
                msg = "Invalid or expired access token"
                raise AuthenticationError(message=msg)
            if response.status_code == 403:
                msg = f"No access to provider: {provider}"
                raise AuthenticationError(message=msg)
            if response.status_code == 404:
                msg = f"Provider key not found: {provider}"
                raise ProviderKeyNotFoundError(message=msg)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            msg = f"Failed to fetch provider key: {e}"
            raise PlatformError(message=msg) from e
        except httpx.RequestError as e:
            msg = f"Network error fetching provider key: {e}"
            raise PlatformError(message=msg) from e

        # Decrypt the provider key
        components = parse_any_llm_key(any_llm_key)
        private_key = load_private_key(components.base64_encoded_private_key)
        api_key = decrypt_sealed_box(data["encrypted_key"], private_key)

        return DecryptedProviderKey(
            api_key=api_key,
            provider_key_id=UUID(data["id"]),
            project_id=UUID(data["project_id"]),
            provider=data["provider"],
            created_at=_parse_iso8601(data["created_at"]),  # type: ignore[arg-type]
            updated_at=_parse_iso8601(data.get("updated_at")),
        )

    async def post_usage_event(
        self,
        any_llm_key: str,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        provider_key_id: UUID,
        project_id: UUID,
        client_name: str,
        duration_ms: float,
        timestamp: datetime | None = None,
        stream: bool = False,
        time_to_first_token_ms: float | None = None,
        tokens_per_second: float | None = None,
    ) -> None:
        """Post a usage event to the platform.

        Args:
            any_llm_key: The ANY_LLM_KEY string
            provider: Provider name
            model: Model name
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            provider_key_id: Provider key UUID
            project_id: Project UUID
            client_name: Client identifier
            duration_ms: Request duration in milliseconds
            timestamp: Event timestamp (defaults to now)
            stream: Whether this was a streaming request
            time_to_first_token_ms: Time to first token in ms (streaming only)
            tokens_per_second: Throughput metric (streaming only)

        """
        access_token = await self.ensure_valid_token(any_llm_key)
        client = await self._get_client()
        url = f"{self.base_url}/usage-events"

        if timestamp is None:
            timestamp = datetime.now(tz=UTC)

        payload: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "provider_key_id": str(provider_key_id),
            "project_id": str(project_id),
            "client_name": client_name,
            "duration_ms": duration_ms,
            "timestamp": timestamp.isoformat(),
        }

        if stream:
            payload["stream"] = True
            if time_to_first_token_ms is not None:
                payload["time_to_first_token_ms"] = time_to_first_token_ms
            if tokens_per_second is not None:
                payload["tokens_per_second"] = tokens_per_second

        try:
            response = await client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if response.status_code not in (200, 201):
                logger.warning(
                    "Failed to post usage event: %s %s",
                    response.status_code,
                    response.text,
                )
        except Exception:
            # Usage tracking should not block the main request
            logger.exception("Error posting usage event")


def is_platform_key(api_key: str | None) -> bool:
    """Check if the provided key is a platform key.

    Args:
        api_key: The API key to check

    Returns:
        True if the key matches the ANY_LLM_KEY format

    """
    if api_key is None:
        return False
    return ANY_LLM_KEY_PATTERN.match(api_key) is not None


class PlatformProvider(AnyLLM):
    """Platform provider that wraps other providers with key management and usage tracking.

    This provider acts as a transparent proxy, exchanging platform keys for
    provider-specific credentials and tracking usage metrics.
    """

    _platform_client: PlatformClient
    _any_llm_key: str
    _provider: AnyLLM | None
    _provider_key_id: UUID | None
    _project_id: UUID | None
    _client_name: str
    _api_base: str | None
    _kwargs: dict[str, Any]

    def _init_client(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the platform provider.

        Args:
            api_key: The ANY_LLM_KEY for platform authentication.
            api_base: Base URL for the provider API (not the platform API).
            **kwargs: Additional arguments including:
                - platform_base_url: Base URL for the platform API
                - client_name: Client identifier for usage tracking

        Raises:
            InvalidKeyFormatError: If the api_key is not a valid platform key.

        """
        if api_key is None:
            import os

            api_key = os.environ.get(ENV_VAR_NAME)

        if api_key is None:
            msg = f"No {ENV_VAR_NAME} provided"
            raise InvalidKeyFormatError(message=msg)

        if not is_platform_key(api_key):
            raise InvalidKeyFormatError

        platform_base_url = kwargs.pop("platform_base_url", DEFAULT_PLATFORM_BASE_URL)
        self._client_name = kwargs.pop("client_name", "any-llm-python")

        self._any_llm_key = api_key
        self._api_base = api_base
        self._kwargs = kwargs
        self._platform_client = PlatformClient(platform_base_url)
        self._provider = None
        self._provider_key_id = None
        self._project_id = None

    async def _ensure_provider(self, provider_name: str) -> AnyLLM:
        """Ensure the wrapped provider is initialized.

        Args:
            provider_name: The provider name (e.g., "openai", "anthropic")

        Returns:
            The initialized provider instance

        """
        if self._provider is None:
            # Get decrypted provider key
            result = await self._platform_client.get_provider_key(
                self._any_llm_key, provider_name
            )

            # Store tracking identifiers
            self._provider_key_id = result.provider_key_id
            self._project_id = result.project_id

            # Import provider class dynamically
            from any_llm.providers import get_provider_class

            provider_class = get_provider_class(provider_name)

            # Create provider instance with decrypted key
            self._provider = provider_class()
            self._provider._init_client(
                api_key=result.api_key,
                api_base=self._api_base,
                **self._kwargs,
            )

        return self._provider

    async def _track_usage(
        self,
        provider: str,
        model: str,
        response: ChatCompletion,
        duration_ms: float,
        stream: bool = False,
        time_to_first_token_ms: float | None = None,
        tokens_per_second: float | None = None,
    ) -> None:
        """Track usage for a completion request.

        Args:
            provider: Provider name
            model: Model name
            response: The completion response
            duration_ms: Request duration in milliseconds
            stream: Whether this was a streaming request
            time_to_first_token_ms: Time to first token (streaming)
            tokens_per_second: Throughput metric (streaming)

        """
        if (
            self._provider_key_id is None
            or self._project_id is None
            or response.usage is None
        ):
            return

        try:
            await self._platform_client.post_usage_event(
                self._any_llm_key,
                provider=provider,
                model=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                provider_key_id=self._provider_key_id,
                project_id=self._project_id,
                client_name=self._client_name,
                duration_ms=duration_ms,
                stream=stream,
                time_to_first_token_ms=time_to_first_token_ms,
                tokens_per_second=tokens_per_second,
            )
        except Exception:
            # Usage tracking should not block the main request
            logger.exception("Failed to track usage")

    async def acompletion(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        *,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        presence_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        user: str | None = None,
        provider: str = "openai",
        **kwargs: Any,
    ) -> ChatCompletion:
        """Perform a chat completion request through the platform.

        Args:
            model: Model ID to use for the completion.
            messages: List of messages comprising the conversation.
            frequency_penalty: Penalize repeated tokens (-2.0 to 2.0).
            logit_bias: Map of token IDs to bias values.
            logprobs: Whether to return log probabilities.
            max_completion_tokens: Maximum tokens to generate.
            max_tokens: Legacy parameter for maximum tokens.
            n: Number of completions to generate.
            parallel_tool_calls: Enable parallel function calling.
            presence_penalty: Penalize new topics (-2.0 to 2.0).
            response_format: Format specification for structured outputs.
            seed: Seed for deterministic sampling.
            stop: Stop sequences.
            stream_options: Options for streaming.
            temperature: Sampling temperature (0 to 2).
            tool_choice: Controls tool calling behavior.
            tools: List of tools the model may call.
            top_logprobs: Number of top tokens to return.
            top_p: Nucleus sampling threshold.
            user: Unique user identifier.
            provider: The underlying provider to use (e.g., "openai", "anthropic").
            **kwargs: Additional provider-specific arguments.

        Returns:
            ChatCompletion response object.

        """
        import time

        start_time = time.time()

        # Ensure provider is initialized
        wrapped_provider = await self._ensure_provider(provider)

        # Delegate to wrapped provider
        response = await wrapped_provider.acompletion(
            model=model,
            messages=messages,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream_options=stream_options,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_logprobs=top_logprobs,
            top_p=top_p,
            user=user,
            **kwargs,
        )

        duration_ms = (time.time() - start_time) * 1000

        # Track usage (fire-and-forget)
        await self._track_usage(provider, model, response, duration_ms)

        return response

    async def acompletion_stream(
        self,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        *,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, int] | None = None,
        logprobs: bool | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        presence_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        user: str | None = None,
        provider: str = "openai",
        **kwargs: Any,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Perform a streaming chat completion request through the platform.

        Args:
            model: Model ID to use for the completion.
            messages: List of messages comprising the conversation.
            frequency_penalty: Penalize repeated tokens (-2.0 to 2.0).
            logit_bias: Map of token IDs to bias values.
            logprobs: Whether to return log probabilities.
            max_completion_tokens: Maximum tokens to generate.
            max_tokens: Legacy parameter for maximum tokens.
            n: Number of completions to generate.
            parallel_tool_calls: Enable parallel function calling.
            presence_penalty: Penalize new topics (-2.0 to 2.0).
            response_format: Format specification for structured outputs.
            seed: Seed for deterministic sampling.
            stop: Stop sequences.
            stream_options: Options for streaming.
            temperature: Sampling temperature (0 to 2).
            tool_choice: Controls tool calling behavior.
            tools: List of tools the model may call.
            top_logprobs: Number of top tokens to return.
            top_p: Nucleus sampling threshold.
            user: Unique user identifier.
            provider: The underlying provider to use (e.g., "openai", "anthropic").
            **kwargs: Additional provider-specific arguments.

        Yields:
            ChatCompletionChunk objects.

        """
        import time

        start_time = time.time()
        first_token_time: float | None = None
        last_chunk: ChatCompletionChunk | None = None

        # Ensure provider is initialized
        wrapped_provider = await self._ensure_provider(provider)

        # Get the stream from wrapped provider
        stream = wrapped_provider.acompletion_stream(
            model=model,
            messages=messages,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream_options=stream_options,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_logprobs=top_logprobs,
            top_p=top_p,
            user=user,
            **kwargs,
        )

        async for chunk in stream:
            # Record time of first content token
            if first_token_time is None:
                if chunk.choices and chunk.choices[0].delta.content:
                    first_token_time = time.time()

            last_chunk = chunk
            yield chunk

        # After stream completes, track usage if available
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        if (
            last_chunk is not None
            and hasattr(last_chunk, "usage")
            and last_chunk.usage is not None
            and self._provider_key_id is not None
            and self._project_id is not None
        ):
            # Calculate streaming metrics
            time_to_first_token_ms = None
            tokens_per_second = None

            if first_token_time is not None:
                time_to_first_token_ms = (first_token_time - start_time) * 1000

            duration_seconds = end_time - start_time
            if duration_seconds > 0:
                total_tokens = (
                    last_chunk.usage.prompt_tokens + last_chunk.usage.completion_tokens
                )
                tokens_per_second = total_tokens / duration_seconds

            # Build a minimal ChatCompletion for tracking
            tracking_response = ChatCompletion(
                id=last_chunk.id,
                model=last_chunk.model,
                object="chat.completion",
                created=last_chunk.created,
                choices=[],
                usage=last_chunk.usage,
            )

            try:
                await self._track_usage(
                    provider,
                    model,
                    tracking_response,
                    duration_ms,
                    stream=True,
                    time_to_first_token_ms=time_to_first_token_ms,
                    tokens_per_second=tokens_per_second,
                )
            except Exception:
                logger.exception("Failed to track streaming usage")

    @staticmethod
    def _convert_completion_params(
        params: CompletionParams, **kwargs: Any
    ) -> dict[str, Any]:
        """Convert completion parameters to provider-specific format.

        Note: PlatformProvider delegates this to the wrapped provider.

        Args:
            params: The completion parameters.
            **kwargs: Additional arguments.

        Returns:
            Provider-specific parameters dictionary.

        """
        return params.to_api_params()

    @staticmethod
    def _convert_completion_response(response: Any) -> ChatCompletion:
        """Convert provider response to ChatCompletion format.

        Note: PlatformProvider delegates this to the wrapped provider.

        Args:
            response: The provider-specific response.

        Returns:
            ChatCompletion object.

        """
        # The wrapped provider handles conversion
        return response  # type: ignore[no-any-return]

    @staticmethod
    def _convert_completion_chunk_response(
        response: Any, **kwargs: Any
    ) -> ChatCompletionChunk:
        """Convert provider streaming response chunk to ChatCompletionChunk format.

        Note: PlatformProvider delegates this to the wrapped provider.

        Args:
            response: The provider-specific chunk response.
            **kwargs: Additional arguments.

        Returns:
            ChatCompletionChunk object.

        """
        # The wrapped provider handles conversion
        return response  # type: ignore[no-any-return]

    async def close(self) -> None:
        """Close the platform client connection."""
        await self._platform_client.close()
