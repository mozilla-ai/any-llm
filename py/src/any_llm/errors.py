class AnyLLMError(Exception):
    """Base exception class for any-llm errors."""

    default_message: str = "An error occurred"

    def __init__(
        self,
        message: str | None = None,
        original_exception: Exception | None = None,
        provider_name: str | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            message: Error message. If None, uses the default message.
            original_exception: The original exception that caused this error.
            provider_name: Name of the provider that raised this error.

        """
        self.message = message or self.default_message
        super().__init__(self.message)
        self.original_exception = original_exception
        self.provider_name = provider_name

    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.provider_name:
            return f"[{self.provider_name}] {self.message}"
        return self.message


class RateLimitError(AnyLLMError):
    """Raised when the API rate limit is exceeded."""

    default_message = "Rate limit exceeded"


class AuthenticationError(AnyLLMError):
    """Raised when authentication with the provider fails."""

    default_message = "Authentication failed"


class InvalidRequestError(AnyLLMError):
    """Raised when the request to the provider is invalid."""

    default_message = "Invalid request"


class ProviderError(AnyLLMError):
    """Raised when the provider encounters an internal error."""

    default_message = "Provider error"


class ContentFilterError(AnyLLMError):
    """Raised when content is blocked by the provider's safety filter."""

    default_message = "Content blocked by safety filter"


class ModelNotFoundError(AnyLLMError):
    """Raised when the requested model is not found or not available."""

    default_message = "Model not found"


class ContextLengthExceededError(AnyLLMError):
    """Raised when the input exceeds the model's maximum context length."""

    default_message = "Context length exceeded"


class MissingApiKeyError(AnyLLMError):
    """Raised when a required API key is not provided."""

    def __init__(self, provider_name: str, env_var_name: str) -> None:
        """Initialize the error.

        Args:
            provider_name: Name of the provider requiring the API key.
            env_var_name: Name of the environment variable for the API key.

        """
        self.env_var_name = env_var_name
        message = (
            f"No {provider_name} API key provided. "
            f"Please provide it in the config or set the {env_var_name} environment variable."
        )
        super().__init__(message, provider_name=provider_name)
