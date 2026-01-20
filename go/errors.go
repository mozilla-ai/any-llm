// Package anyllm provides a unified interface for interacting with multiple LLM providers.
package anyllm

import "fmt"

// AnyLLMError is the base error type for all any-llm errors.
// It preserves the original exception from the provider SDK.
type AnyLLMError struct {
	Message           string
	OriginalException error
	ProviderName      string
}

// Error implements the error interface.
func (e *AnyLLMError) Error() string {
	if e.ProviderName != "" {
		return fmt.Sprintf("[%s] %s", e.ProviderName, e.Message)
	}
	return e.Message
}

// Unwrap returns the original exception.
func (e *AnyLLMError) Unwrap() error {
	return e.OriginalException
}

// NewAnyLLMError creates a new AnyLLMError with the given message.
func NewAnyLLMError(message string, originalException error, providerName string) *AnyLLMError {
	if message == "" {
		message = "An error occurred"
	}
	return &AnyLLMError{
		Message:           message,
		OriginalException: originalException,
		ProviderName:      providerName,
	}
}

// RateLimitError is raised when the API rate limit is exceeded.
type RateLimitError struct {
	AnyLLMError
}

// NewRateLimitError creates a new RateLimitError.
func NewRateLimitError(message string, originalException error, providerName string) *RateLimitError {
	if message == "" {
		message = "Rate limit exceeded"
	}
	return &RateLimitError{
		AnyLLMError: AnyLLMError{
			Message:           message,
			OriginalException: originalException,
			ProviderName:      providerName,
		},
	}
}

// AuthenticationError is raised when authentication with the provider fails.
type AuthenticationError struct {
	AnyLLMError
}

// NewAuthenticationError creates a new AuthenticationError.
func NewAuthenticationError(message string, originalException error, providerName string) *AuthenticationError {
	if message == "" {
		message = "Authentication failed"
	}
	return &AuthenticationError{
		AnyLLMError: AnyLLMError{
			Message:           message,
			OriginalException: originalException,
			ProviderName:      providerName,
		},
	}
}

// InvalidRequestError is raised when the request to the provider is invalid.
type InvalidRequestError struct {
	AnyLLMError
}

// NewInvalidRequestError creates a new InvalidRequestError.
func NewInvalidRequestError(message string, originalException error, providerName string) *InvalidRequestError {
	if message == "" {
		message = "Invalid request"
	}
	return &InvalidRequestError{
		AnyLLMError: AnyLLMError{
			Message:           message,
			OriginalException: originalException,
			ProviderName:      providerName,
		},
	}
}

// ProviderError is raised when the provider encounters an internal error.
type ProviderError struct {
	AnyLLMError
}

// NewProviderError creates a new ProviderError.
func NewProviderError(message string, originalException error, providerName string) *ProviderError {
	if message == "" {
		message = "Provider error"
	}
	return &ProviderError{
		AnyLLMError: AnyLLMError{
			Message:           message,
			OriginalException: originalException,
			ProviderName:      providerName,
		},
	}
}

// ContentFilterError is raised when content is blocked by the provider's safety filter.
type ContentFilterError struct {
	AnyLLMError
}

// NewContentFilterError creates a new ContentFilterError.
func NewContentFilterError(message string, originalException error, providerName string) *ContentFilterError {
	if message == "" {
		message = "Content blocked by safety filter"
	}
	return &ContentFilterError{
		AnyLLMError: AnyLLMError{
			Message:           message,
			OriginalException: originalException,
			ProviderName:      providerName,
		},
	}
}

// ModelNotFoundError is raised when the requested model is not found or not available.
type ModelNotFoundError struct {
	AnyLLMError
}

// NewModelNotFoundError creates a new ModelNotFoundError.
func NewModelNotFoundError(message string, originalException error, providerName string) *ModelNotFoundError {
	if message == "" {
		message = "Model not found"
	}
	return &ModelNotFoundError{
		AnyLLMError: AnyLLMError{
			Message:           message,
			OriginalException: originalException,
			ProviderName:      providerName,
		},
	}
}

// ContextLengthExceededError is raised when the input exceeds the model's maximum context length.
type ContextLengthExceededError struct {
	AnyLLMError
}

// NewContextLengthExceededError creates a new ContextLengthExceededError.
func NewContextLengthExceededError(message string, originalException error, providerName string) *ContextLengthExceededError {
	if message == "" {
		message = "Context length exceeded"
	}
	return &ContextLengthExceededError{
		AnyLLMError: AnyLLMError{
			Message:           message,
			OriginalException: originalException,
			ProviderName:      providerName,
		},
	}
}

// MissingAPIKeyError is raised when a required API key is not provided.
type MissingAPIKeyError struct {
	AnyLLMError
	EnvVarName string
}

// NewMissingAPIKeyError creates a new MissingAPIKeyError.
func NewMissingAPIKeyError(providerName, envVarName string) *MissingAPIKeyError {
	message := fmt.Sprintf(
		"No %s API key provided. Please provide it in the config or set the %s environment variable.",
		providerName,
		envVarName,
	)
	return &MissingAPIKeyError{
		AnyLLMError: AnyLLMError{
			Message:      message,
			ProviderName: providerName,
		},
		EnvVarName: envVarName,
	}
}
