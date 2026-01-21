//! Error types for any-llm.
//!
//! This module provides unified error types for LLM provider interactions.

use thiserror::Error;

/// Base error type for any-llm operations.
#[derive(Error, Debug)]
pub enum AnyLLMError {
    /// Rate limit exceeded error.
    #[error("[{provider_name}] Rate limit exceeded")]
    RateLimit {
        /// Name of the provider that raised the error.
        provider_name: String,
        /// Original error message from the provider.
        message: Option<String>,
    },

    /// Authentication failed error.
    #[error("[{provider_name}] Authentication failed")]
    Authentication {
        /// Name of the provider that raised the error.
        provider_name: String,
        /// Original error message from the provider.
        message: Option<String>,
    },

    /// Invalid request error.
    #[error("[{provider_name}] Invalid request: {message}")]
    InvalidRequest {
        /// Name of the provider that raised the error.
        provider_name: String,
        /// Error message describing what was invalid.
        message: String,
    },

    /// Provider internal error.
    #[error("[{provider_name}] Provider error")]
    ProviderError {
        /// Name of the provider that raised the error.
        provider_name: String,
        /// Original error message from the provider.
        message: Option<String>,
    },

    /// Content blocked by safety filter.
    #[error("[{provider_name}] Content blocked by safety filter")]
    ContentFilter {
        /// Name of the provider that raised the error.
        provider_name: String,
        /// Original error message from the provider.
        message: Option<String>,
    },

    /// Model not found error.
    #[error("[{provider_name}] Model not found")]
    ModelNotFound {
        /// Name of the provider that raised the error.
        provider_name: String,
        /// Original error message from the provider.
        message: Option<String>,
    },

    /// Context length exceeded error.
    #[error("[{provider_name}] Context length exceeded")]
    ContextLengthExceeded {
        /// Name of the provider that raised the error.
        provider_name: String,
        /// Original error message from the provider.
        message: Option<String>,
    },

    /// Missing API key error.
    #[error(
        "No {provider_name} API key provided. Please provide it in the config or set the {env_var_name} environment variable."
    )]
    MissingApiKey {
        /// Name of the provider.
        provider_name: String,
        /// Name of the environment variable that should be set.
        env_var_name: String,
    },

    /// HTTP client error.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON parsing error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Stream parsing error.
    #[error("Stream error: {0}")]
    Stream(String),
}

impl AnyLLMError {
    /// Create a rate limit error.
    #[must_use]
    pub fn rate_limit(provider_name: impl Into<String>, message: Option<String>) -> Self {
        Self::RateLimit {
            provider_name: provider_name.into(),
            message,
        }
    }

    /// Create an authentication error.
    #[must_use]
    pub fn authentication(provider_name: impl Into<String>, message: Option<String>) -> Self {
        Self::Authentication {
            provider_name: provider_name.into(),
            message,
        }
    }

    /// Create an invalid request error.
    #[must_use]
    pub fn invalid_request(provider_name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            provider_name: provider_name.into(),
            message: message.into(),
        }
    }

    /// Create a provider error.
    #[must_use]
    pub fn provider_error(provider_name: impl Into<String>, message: Option<String>) -> Self {
        Self::ProviderError {
            provider_name: provider_name.into(),
            message,
        }
    }

    /// Create a content filter error.
    #[must_use]
    pub fn content_filter(provider_name: impl Into<String>, message: Option<String>) -> Self {
        Self::ContentFilter {
            provider_name: provider_name.into(),
            message,
        }
    }

    /// Create a model not found error.
    #[must_use]
    pub fn model_not_found(provider_name: impl Into<String>, message: Option<String>) -> Self {
        Self::ModelNotFound {
            provider_name: provider_name.into(),
            message,
        }
    }

    /// Create a context length exceeded error.
    #[must_use]
    pub fn context_length_exceeded(
        provider_name: impl Into<String>,
        message: Option<String>,
    ) -> Self {
        Self::ContextLengthExceeded {
            provider_name: provider_name.into(),
            message,
        }
    }

    /// Create a missing API key error.
    #[must_use]
    pub fn missing_api_key(
        provider_name: impl Into<String>,
        env_var_name: impl Into<String>,
    ) -> Self {
        Self::MissingApiKey {
            provider_name: provider_name.into(),
            env_var_name: env_var_name.into(),
        }
    }

    /// Create a stream error.
    #[must_use]
    pub fn stream(message: impl Into<String>) -> Self {
        Self::Stream(message.into())
    }
}

/// Result type alias for any-llm operations.
pub type Result<T> = std::result::Result<T, AnyLLMError>;
