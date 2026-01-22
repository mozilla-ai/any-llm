# Platform Provider Integration Schema

The **Platform Provider** is a transparent proxy layer that wraps any existing provider to add centralized key management and usage tracking. Users authenticate with a single platform key (`ANY_LLM_KEY`), which the platform exchanges for provider-specific credentials at runtime.

## Overview

This document provides a comprehensive specification for implementing the ANY LLM Platform Client in any programming language. It covers:
- Authentication flow (challenge-response with X25519 encryption)
- API endpoints and data structures
- Cryptographic operations
- Token management
- Error handling

## Key Components

### 1. Platform Key Format

Platform keys follow a specific format that distinguishes them from regular provider keys:

```
ANY.v<version>.<kid>.<fingerprint>-<base64_key>

Example: ANY.v1.abc123.xyz789-SGVsbG9Xb3JsZA==
```

**Validation regex:** `^ANY\.v\d+\.([^.]+)\.([^-]+)-(.+)$`

**Components:**
- `ANY.v<version>`: Protocol identifier and version (currently "ANY.v1")
- `<kid>`: Key identifier (unique identifier for this key)
- `<fingerprint>`: Public key fingerprint for validation
- `<base64_key>`: Base64-encoded 32-byte X25519 private key

**Parsing:**
```python
import re

def parse_any_llm_key(any_llm_key: str) -> tuple:
    """Parse ANY_LLM_KEY into components.
    
    Returns: (key_id, public_key_fingerprint, base64_encoded_private_key)
    Raises: ValueError if format is invalid
    """
    match = re.match(r"^ANY\.v1\.([^.]+)\.([^-]+)-(.+)$", any_llm_key)
    if not match:
        raise ValueError("Invalid ANY_LLM_KEY format")
    return match.groups()  # (key_id, fingerprint, private_key_base64)
```

```

### 2. Cryptographic Operations

The platform uses **X25519 sealed box encryption** with XChaCha20-Poly1305 AEAD.

#### Key Management

**Load Private Key:**
```python
import base64

def load_private_key(private_key_base64: str) -> bytes:
    """Load X25519 private key from base64.
    
    Args:
        private_key_base64: Base64-encoded 32-byte private key
    
    Returns: 32-byte private key
    Raises: ValueError if not exactly 32 bytes
    """
    private_key_bytes = base64.b64decode(private_key_base64)
    if len(private_key_bytes) != 32:
        raise ValueError(f"X25519 private key must be 32 bytes, got {len(private_key_bytes)}")
    return private_key_bytes
```

**Extract Public Key:**
```python
def extract_public_key(private_key: bytes) -> str:
    """Derive public key from X25519 private key.
    
    Uses Curve25519 scalar multiplication: public = base_point * private
    
    Args:
        private_key: 32-byte X25519 private key
    
    Returns: Base64-encoded 32-byte public key
    """
    # Use your language's X25519/Curve25519 library
    # Python example with PyNaCl:
    # public_key_bytes = nacl.public.PrivateKey(private_key).public_key
    public_key_bytes = curve25519_scalar_mult_base(private_key)
    return base64.b64encode(public_key_bytes).decode('utf-8')
```

#### Sealed Box Decryption

Sealed boxes use an **ephemeral sender key** for anonymous encryption. The decryption process:

1. Extract ephemeral public key (first 32 bytes)
2. Extract ciphertext (remaining bytes)
3. Derive shared secret via X25519
4. Generate nonce from SHA-512(ephemeral_pk || recipient_pk)[:24]
5. Decrypt using XChaCha20-Poly1305 AEAD

```python
import hashlib

def decrypt_sealed_box(encrypted_data_base64: str, private_key: bytes) -> str:
    """Decrypt X25519 sealed box data.
    
    Format: [ephemeral_public_key(32)][ciphertext(N+16)]
    
    Args:
        encrypted_data_base64: Base64-encoded sealed box
        private_key: 32-byte X25519 private key
    
    Returns: Decrypted UTF-8 string
    Raises: ValueError if format invalid or decryption fails
    """
    encrypted_data = base64.b64decode(encrypted_data_base64)
    
    if len(encrypted_data) < 32:
        raise ValueError("Invalid sealed box: too short")
    
    # Split sealed box components
    ephemeral_public_key = encrypted_data[:32]
    ciphertext = encrypted_data[32:]
    
    # Derive recipient public key from private key
    recipient_public_key = curve25519_scalar_mult_base(private_key)
    
    # Compute shared secret: shared = ephemeral_pk * private_key
    shared_secret = curve25519_scalar_mult(private_key, ephemeral_public_key)
    
    # Generate nonce: SHA-512(ephemeral_pk || recipient_pk)[:24]
    combined = ephemeral_public_key + recipient_public_key
    nonce = hashlib.sha512(combined).digest()[:24]
    
    # Decrypt with XChaCha20-Poly1305
    plaintext = xchacha20poly1305_decrypt(ciphertext, nonce, shared_secret)
    
    return plaintext.decode('utf-8')
```

**Cryptographic Primitives Required:**
- **X25519**: Elliptic curve Diffie-Hellman (ECDH) on Curve25519
- **XChaCha20-Poly1305**: Authenticated encryption with associated data (AEAD)
- **SHA-512**: For nonce derivation

**Recommended Libraries by Language:**
- Python: `PyNaCl` (libsodium bindings)
- JavaScript/Node: `tweetnacl`, `libsodium-wrappers`
- Go: `golang.org/x/crypto/nacl/box`, `golang.org/x/crypto/curve25519`
- Rust: `crypto_box`, `x25519-dalek`
- Java: `lazysodium-java` (libsodium bindings)
- C#: `NSec`, `Sodium.Core`

### 3. Authentication Flow (Challenge-Response)

The platform uses a secure challenge-response protocol to authenticate clients without transmitting the private key.

#### Flow Overview

```
Client                          Platform API
  |                                  |
  | 1. POST /auth/                   |
  |    {encryption_key: <public>}    |
  |---------------------------------→|
  |                                  |
  | 2. 200 OK                        |
  |    {encrypted_challenge: <enc>}  |
  |←---------------------------------|
  |                                  |
  | 3. Decrypt challenge locally     |
  |    (yields UUID)                 |
  |                                  |
  | 4. POST /auth/token              |
  |    {solved_challenge: <UUID>}    |
  |---------------------------------→|
  |                                  |
  | 5. 200 OK                        |
  |    {access_token: <JWT>,         |
  |     token_type: "bearer",        |
  |     expires_in: 86400}           |
  |←---------------------------------|
```

#### Step-by-Step Implementation

**Step 1: Create Challenge**

```http
POST /api/v1/auth/
Content-Type: application/json

{
  "encryption_key": "<base64_public_key>"
}
```

**Response (200 OK):**
```json
{
  "encrypted_challenge": "<base64_encrypted_uuid>"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid public key format
- `404 Not Found`: No project found for this public key
- `500 Internal Server Error`: Server-side error

**Step 2: Solve Challenge**

Decrypt the encrypted challenge using your private key:

```python
def solve_challenge(encrypted_challenge: str, private_key: bytes) -> str:
    """Decrypt and solve authentication challenge.
    
    Args:
        encrypted_challenge: Base64-encoded sealed box containing UUID
        private_key: 32-byte X25519 private key
    
    Returns: UUID string (e.g., "550e8400-e29b-41d4-a716-446655440000")
    """
    decrypted_uuid_str = decrypt_sealed_box(encrypted_challenge, private_key)
    # Validate UUID format
    uuid_obj = parse_uuid(decrypted_uuid_str)
    return str(uuid_obj)
```

**Step 3: Request Access Token**

```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "solved_challenge": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response (200 OK):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

**Error Responses:**
- `400 Bad Request`: Invalid challenge format
- `401 Unauthorized`: Challenge not found or expired
- `500 Internal Server Error`: Server-side error

**Token Management:**
- Tokens expire after 24 hours (86400 seconds)
- Recommended: Store token and refresh 1 hour before expiration
- Tokens are JWTs but should be treated as opaque strings by clients

### 4. Provider Key Retrieval

After obtaining an access token, fetch the encrypted provider key.

**Request:**
```http
GET /api/v1/provider-keys/{provider}
Authorization: Bearer <access_token>
```

**Path Parameters:**
- `provider`: Provider name (e.g., `openai`, `anthropic`, `google`, `cohere`)

**Response (200 OK):**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "project_id": "987fcdeb-51a2-43f1-b456-426614174001",
  "provider": "openai",
  "encrypted_key": "<base64_encrypted_api_key>",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-20T14:45:00Z"
}
```

**Error Responses:**
- `401 Unauthorized`: Missing or invalid access token
- `403 Forbidden`: Valid token but no access to this provider
- `404 Not Found`: Provider key not found for this project
- `500 Internal Server Error`: Server-side error

**Decrypt Provider Key:**
```python
def decrypt_provider_key(encrypted_key: str, private_key: bytes) -> str:
    """Decrypt provider API key.
    
    Args:
        encrypted_key: Base64-encoded sealed box from API response
        private_key: 32-byte X25519 private key
    
    Returns: Decrypted provider API key (e.g., "sk-...")
    """
    return decrypt_sealed_box(encrypted_key, private_key)
```

### 5. Complete Client Implementation

**High-Level Convenience Method:**

Most implementations should provide a simple one-call method:

```python
def get_decrypted_provider_key(
    any_llm_key: str,
    provider: str,
    base_url: str = "http://localhost:8000/api/v1"
) -> dict:
    """Get decrypted provider key with metadata (recommended method).
    
    Handles complete flow:
    1. Parse ANY_LLM_KEY
    2. Authenticate and get access token (cached)
    3. Fetch encrypted provider key
    4. Decrypt and return with metadata
    
    Args:
        any_llm_key: ANY_LLM_KEY string
        provider: Provider name (e.g., "openai")
        base_url: Platform API base URL
    
    Returns:
        {
            "api_key": "sk-...",
            "provider_key_id": UUID,
            "project_id": UUID,
            "provider": "openai",
            "created_at": datetime,
            "updated_at": datetime | None
        }
    """
    # 1. Parse key
    key_id, fingerprint, private_key_b64 = parse_any_llm_key(any_llm_key)
    private_key = load_private_key(private_key_b64)
    public_key = extract_public_key(private_key)
    
    # 2. Get access token (check cache first)
    access_token = ensure_valid_token(any_llm_key, base_url)
    
    # 3. Fetch encrypted provider key
    response = http_get(
        f"{base_url}/provider-keys/{provider}",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    provider_data = response.json()
    
    # 4. Decrypt and return
    api_key = decrypt_sealed_box(provider_data["encrypted_key"], private_key)
    
    return {
        "api_key": api_key,
        "provider_key_id": UUID(provider_data["id"]),
        "project_id": UUID(provider_data["project_id"]),
        "provider": provider_data["provider"],
        "created_at": parse_iso8601(provider_data["created_at"]),
        "updated_at": parse_iso8601(provider_data.get("updated_at"))
    }
```

**Token Caching:**

```python
class PlatformClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.access_token = None
        self.token_expires_at = None
    
    def ensure_valid_token(self, any_llm_key: str) -> str:
        """Get valid access token, refreshing if needed.
        
        Returns: Valid access token string
        """
        now = current_time()
        
        # Check if we need a new token
        if not self.access_token or not self.token_expires_at or now >= self.token_expires_at:
            self.refresh_access_token(any_llm_key)
        
        return self.access_token
    
    def refresh_access_token(self, any_llm_key: str) -> str:
        """Authenticate and get new access token.
        
        Returns: New access token string
        """
        # Parse key
        _, _, private_key_b64 = parse_any_llm_key(any_llm_key)
        private_key = load_private_key(private_key_b64)
        public_key = extract_public_key(private_key)
        
        # Create challenge
        response = http_post(
            f"{self.base_url}/auth/",
            json={"encryption_key": public_key}
        )
        encrypted_challenge = response.json()["encrypted_challenge"]
        
        # Solve challenge
        solved_uuid = decrypt_sealed_box(encrypted_challenge, private_key)
        
        # Get token
        response = http_post(
            f"{self.base_url}/auth/token",
            json={"solved_challenge": solved_uuid}
        )
        token_data = response.json()
        
        # Cache token with safety margin (23 hours instead of 24)
        self.access_token = token_data["access_token"]
        self.token_expires_at = now + timedelta(hours=23)
        
        return self.access_token
```

### 6. Provider Resolution Flow

When `AnyLLM.create(provider, api_key)` is called:

1. Load the requested provider class (e.g., `OpenaiProvider`)
2. Check if `api_key` matches the platform key format
3. **If platform key:**
   - Instantiate `PlatformProvider` with the platform key
   - Set `platform_provider.provider = <ProviderClass>` (triggers key exchange)
   - Return the platform provider instance
4. **If regular key:** Return the provider instance directly

### 7. Provider Setter (Key Exchange)

When `provider` property is set:

```python
def set_provider(provider_class):
    """Exchange platform key for provider-specific API key.
    
    This method is called when the provider is set on PlatformProvider.
    It fetches and decrypts the actual provider API key.
    """
    # 1. Call platform API to exchange platform key for provider key
    result = platform_client.get_decrypted_provider_key(
        any_llm_key=self.any_llm_key,
        provider=provider_class.PROVIDER_NAME
    )
    
    # 2. Store tracking identifiers for usage reporting
    self.provider_key_id = result.provider_key_id
    self.project_id = result.project_id
    
    # 3. Instantiate real provider with decrypted key
    self._provider = provider_class(
        api_key=result.api_key,
        api_base=self.api_base,
        **self.kwargs
    )
```

### 8. Request Delegation with Usage Tracking

For each API call (completion, embedding, etc.):

```python
async def _acompletion(params):
    start_time = now()
    
    # Delegate to wrapped provider
    response = await self._provider._acompletion(params)
    
    # Post usage event to platform
    await post_usage_event(
        any_llm_key=self.any_llm_key,
        provider=self._provider.PROVIDER_NAME,
        model=response.model,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        provider_key_id=self.provider_key_id,
        client_name=self.client_name,
        duration_ms=(now() - start_time) * 1000
    )
    
    return response
```

### 8. Request Delegation with Usage Tracking

For each API call (completion, embedding, etc.):

```python
async def _acompletion(params):
    """Wrap provider completion with usage tracking.
    
    This method:
    1. Delegates the request to the wrapped provider
    2. Measures execution time
    3. Posts usage metrics to the platform
    """
    start_time = now()
    
    # Delegate to wrapped provider
    response = await self._provider._acompletion(params)
    
    # Post usage event to platform (fire-and-forget or await)
    await post_usage_event(
        any_llm_key=self.any_llm_key,
        provider=self._provider.PROVIDER_NAME,
        model=response.model,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        provider_key_id=self.provider_key_id,
        project_id=self.project_id,
        client_name=self.client_name,
        duration_ms=(now() - start_time) * 1000,
        timestamp=start_time
    )
    
    return response
```

**Usage Event API:**

```http
POST /api/v1/usage-events
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "provider": "openai",
  "model": "gpt-4",
  "input_tokens": 150,
  "output_tokens": 50,
  "provider_key_id": "123e4567-e89b-12d3-a456-426614174000",
  "project_id": "987fcdeb-51a2-43f1-b456-426614174001",
  "client_name": "any-llm-python/1.0.0",
  "duration_ms": 1250.5,
  "timestamp": "2024-01-22T10:30:45.123Z"
}
```

**Response (201 Created):**
```json
{
  "id": "event-uuid",
  "status": "recorded"
}
```

**Error Handling for Usage Events:**
- Usage tracking should NOT block the main request
- Log failures but return the provider response to the user
- Consider retry logic with exponential backoff
- Batch multiple events for efficiency (optional)

### 9. Streaming Considerations

For streaming responses, wrap the iterator to collect metrics:

```python
async def stream_with_tracking(stream):
    """Wrap streaming response to track usage metrics.
    
    Challenges:
    - Token counts not available until stream ends
    - Need to track time to first token (TTFT)
    - Calculate tokens per second throughput
    """
    chunks = []
    first_token_time = None
    start_time = now()
    
    async for chunk in stream:
        # Record time of first content token
        if first_token_time is None and has_content(chunk):
            first_token_time = now()
        
        chunks.append(chunk)
        yield chunk  # Pass through to caller
    
    # After stream completes, extract final metrics
    final_chunk = chunks[-1] if chunks else None
    end_time = now()
    
    # Most providers include usage in the final chunk
    if final_chunk and hasattr(final_chunk, 'usage'):
        usage = final_chunk.usage
        
        # Calculate throughput metrics
        total_tokens = usage.prompt_tokens + usage.completion_tokens
        duration_seconds = (end_time - start_time)
        time_to_first_token_ms = (first_token_time - start_time) * 1000 if first_token_time else None
        tokens_per_second = total_tokens / duration_seconds if duration_seconds > 0 else None
        
        # Post usage event with streaming metrics
        await post_usage_event(
            provider=self._provider.PROVIDER_NAME,
            model=final_chunk.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            provider_key_id=self.provider_key_id,
            project_id=self.project_id,
            duration_ms=duration_seconds * 1000,
            time_to_first_token_ms=time_to_first_token_ms,
            tokens_per_second=tokens_per_second,
            stream=True
        )
```

## API Reference

### Base URL

Default: `http://localhost:8000/api/v1`

Production environments should use HTTPS and proper domain names.

### Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/auth/` | None | Create authentication challenge |
| POST | `/auth/token` | None | Exchange solved challenge for JWT |
| GET | `/provider-keys/{provider}` | Bearer | Fetch encrypted provider key |
| POST | `/usage-events` | Bearer | Report usage metrics (optional) |

### Error Response Format

All errors follow this structure:

```json
{
  "detail": "Human-readable error message",
  "error_code": "ERROR_CODE",
  "status_code": 400
}
```

**Common Error Codes:**
- `INVALID_KEY_FORMAT`: Malformed ANY_LLM_KEY
- `PROJECT_NOT_FOUND`: No project matches the public key
- `CHALLENGE_EXPIRED`: Challenge must be solved within timeout
- `INVALID_TOKEN`: JWT token is invalid or expired
- `PROVIDER_NOT_FOUND`: Requested provider key doesn't exist
- `DECRYPTION_FAILED`: Sealed box decryption failed

## Data Structures

### DecryptedProviderKey

```typescript
interface DecryptedProviderKey {
  api_key: string;           // Decrypted provider API key
  provider_key_id: UUID;     // Unique ID for this provider key
  project_id: UUID;          // Project owning this key
  provider: string;          // Provider name (e.g., "openai")
  created_at: DateTime;      // When key was created
  updated_at: DateTime?;     // Last update timestamp (optional)
}
```

### KeyComponents

```typescript
interface KeyComponents {
  key_id: string;                      // Unique key identifier
  public_key_fingerprint: string;      // Public key fingerprint
  base64_encoded_private_key: string;  // Base64 X25519 private key
}
```

## Implementation Checklist

### Core Requirements

- [ ] Parse `ANY_LLM_KEY` format with regex validation
- [ ] Load 32-byte X25519 private key from base64
- [ ] Derive public key from private key
- [ ] Implement sealed box decryption (X25519 + XChaCha20-Poly1305)
- [ ] HTTP client for API requests (sync and/or async)
- [ ] Challenge creation endpoint (`POST /auth/`)
- [ ] Challenge solving (decrypt UUID)
- [ ] Token request endpoint (`POST /auth/token`)
- [ ] Provider key fetch endpoint (`GET /provider-keys/{provider}`)
- [ ] Decrypt provider API key
- [ ] High-level convenience method: `get_decrypted_provider_key()`

### Recommended Features

- [ ] Token caching with expiration management
- [ ] Automatic token refresh (23-hour safety margin)
- [ ] Async/await support (if language supports it)
- [ ] Proper error handling with specific exception types
- [ ] Logging (debug, info, error levels)
- [ ] Configuration via environment variables
- [ ] CLI tool for interactive key decryption
- [ ] Comprehensive test coverage
- [ ] Type hints/annotations (if language supports)

## Security Considerations

### Key Protection

1. **Never log or print the private key** from `ANY_LLM_KEY`
2. **Treat access tokens as sensitive** - don't log full tokens
3. **Use secure memory** where available (e.g., SecureString in .NET)
4. **Clear sensitive data** after use (overwrite memory if possible)

### Network Security

1. **Use HTTPS** in production (verify TLS certificates)
2. **Validate server certificates** - don't disable SSL verification
3. **Set reasonable timeouts** to prevent hanging requests
4. **Rate limiting** - respect API rate limits

### Cryptographic Best Practices

1. **Use vetted libraries** - don't implement crypto primitives yourself
2. **Validate all inputs** before cryptographic operations
3. **Check sealed box format** before decryption
4. **Handle decryption failures** gracefully

### Error Handling

1. **Don't expose private keys** in error messages
2. **Log securely** - redact sensitive information
3. **Fail securely** - prefer errors over insecure fallbacks

## Testing Guidelines

### Unit Tests

```python
def test_parse_any_llm_key_valid():
    """Test parsing valid ANY_LLM_KEY."""
    key = "ANY.v1.abc123.xyz789-SGVsbG9Xb3JsZA=="
    kid, fingerprint, private_key = parse_any_llm_key(key)
    assert kid == "abc123"
    assert fingerprint == "xyz789"
    assert private_key == "SGVsbG9Xb3JsZA=="

def test_parse_any_llm_key_invalid():
    """Test parsing invalid ANY_LLM_KEY raises ValueError."""
    with pytest.raises(ValueError):
        parse_any_llm_key("INVALID_KEY")

def test_load_private_key_wrong_size():
    """Test loading private key with wrong size raises ValueError."""
    # 16 bytes instead of 32
    with pytest.raises(ValueError):
        load_private_key(base64.b64encode(b"short" * 3).decode())

def test_decrypt_sealed_box_too_short():
    """Test decrypting sealed box that's too short raises ValueError."""
    with pytest.raises(ValueError):
        decrypt_sealed_box(base64.b64encode(b"short").decode(), private_key)
```

### Integration Tests

```python
@pytest.mark.integration
def test_full_authentication_flow(platform_url, test_any_llm_key):
    """Test complete authentication and key retrieval."""
    client = PlatformClient(platform_url)
    result = client.get_decrypted_provider_key(test_any_llm_key, "openai")
    
    assert result["api_key"].startswith("sk-")
    assert isinstance(result["provider_key_id"], UUID)
    assert result["provider"] == "openai"

@pytest.mark.integration
async def test_async_authentication_flow(platform_url, test_any_llm_key):
    """Test async authentication and key retrieval."""
    client = AsyncPlatformClient(platform_url)
    result = await client.aget_decrypted_provider_key(test_any_llm_key, "openai")
    
    assert result["api_key"].startswith("sk-")
```

### Mock Testing

```python
def test_token_caching(mock_http):
    """Test that tokens are cached and reused."""
    client = PlatformClient()
    
    # First call should request token
    client.get_decrypted_provider_key(any_llm_key, "openai")
    assert mock_http.post_count("/auth/token") == 1
    
    # Second call should reuse token
    client.get_decrypted_provider_key(any_llm_key, "anthropic")
    assert mock_http.post_count("/auth/token") == 1  # Still 1, not 2
```

## Language-Specific Notes

### Python
- Use `PyNaCl` for cryptography
- Provide both sync and async interfaces
- Use `httpx` for HTTP (supports both modes)
- Type hints with `typing` module
- CLI with `click` or `argparse`

### JavaScript/TypeScript
- Use `tweetnacl` or `libsodium-wrappers`
- Native async/await with Promises
- `node-fetch` or `axios` for HTTP
- TypeScript interfaces for type safety

### Go
- Use `golang.org/x/crypto/nacl/box`
- Native goroutines for async
- Standard `net/http` package
- Explicit error handling

### Rust
- Use `crypto_box` crate
- Async with `tokio` runtime
- `reqwest` for HTTP client
- Strong type system with `Result<T, E>`

### Java
- Use `lazysodium-java` (libsodium bindings)
- `CompletableFuture` for async
- `OkHttp` or `java.net.http` for HTTP
- Proper exception hierarchy

### C#
- Use `NSec` or `Sodium.Core`
- `async`/`await` with `Task<T>`
- `HttpClient` for HTTP requests
- Record types for data structures

## Troubleshooting

### Common Issues

**"Invalid ANY_LLM_KEY format"**
- Ensure key follows: `ANY.v1.<kid>.<fingerprint>-<base64_key>`
- Check for accidental whitespace or newlines
- Verify base64 encoding is valid

**"X25519 private key must be 32 bytes"**
- Base64-decoded private key must be exactly 32 bytes
- Check base64 encoding/padding

**"No project found for the provided public key"**
- Public key fingerprint doesn't match any project
- Generate a new ANY_LLM_KEY from the platform UI
- Verify you're using the correct environment (dev/prod)

**"Failed to decrypt sealed box"**
- Encrypted data may be corrupted
- Wrong private key used for decryption
- Network error truncated the response

**"Token expired"**
- Implement automatic token refresh
- Check system clock is synchronized
