//! any-llm: A unified interface for interacting with multiple LLM providers.
//!
//! This library provides a common abstraction layer for various LLM providers,
//! allowing you to switch between providers with minimal code changes.
//!
//! # Example
//!
//! ```no_run
//! use any_llm_rs::{AnyLLM, Message, Provider};
//!
//! #[tokio::main]
//! async fn main() -> any_llm_rs::Result<()> {
//!     let client = AnyLLM::create(Provider::OpenAI, None, None, None)?;
//!     
//!     let messages = vec![
//!         Message::user("Hello, how are you?"),
//!     ];
//!     
//!     let response = client.completion("gpt-4o", messages, None).await?;
//!     println!("{:?}", response.choices[0].message.content);
//!     
//!     Ok(())
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod errors;
mod providers;
pub mod types;

use std::collections::HashMap;
use std::pin::Pin;

use futures::Stream;

pub use errors::{AnyLLMError, Result};
use providers::openai::OpenAIProvider;
pub use types::{
    ChatCompletion, ChatCompletionChunk, Choice, ChoiceMessage, CompletionOptions, ContentPart,
    FunctionCall, FunctionDefinition, FunctionTool, ImageContentPart, ImageUrl, JsonSchemaFormat,
    Message, MessageContent, Provider, ResponseFormat, StreamOptions, TextContentPart, ToolCall,
    ToolChoice, ToolChoiceFunction, ToolChoiceObject, Usage,
};

/// Unified LLM client that provides a common interface across providers.
pub struct AnyLLM {
    provider: ProviderClient,
}

enum ProviderClient {
    OpenAI(OpenAIProvider),
}

impl AnyLLM {
    /// Create a new `AnyLLM` client for the specified provider.
    ///
    /// # Arguments
    ///
    /// * `provider` - The LLM provider to use.
    /// * `api_key` - API key for authentication. If None, the provider will
    ///   attempt to read from the standard environment variable.
    /// * `api_base` - Base URL for the API. If None, uses the provider's default.
    /// * `extra_headers` - Additional headers to include in requests.
    ///
    /// # Errors
    ///
    /// Returns an error if the client cannot be initialized (e.g., missing API key).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use any_llm_rs::{AnyLLM, Provider};
    ///
    /// let client = AnyLLM::create(Provider::OpenAI, None, None, None)?;
    /// # Ok::<(), any_llm_rs::AnyLLMError>(())
    /// ```
    pub fn create(
        provider: Provider,
        api_key: Option<String>,
        api_base: Option<String>,
        extra_headers: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let provider_client = match provider {
            Provider::OpenAI => {
                ProviderClient::OpenAI(OpenAIProvider::new(api_key, api_base, extra_headers)?)
            }
        };

        Ok(Self {
            provider: provider_client,
        })
    }

    /// Perform a chat completion request.
    ///
    /// # Arguments
    ///
    /// * `model` - Model ID to use for generation.
    /// * `messages` - The conversation messages.
    /// * `options` - Optional completion parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or the response cannot be parsed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use any_llm_rs::{AnyLLM, Message, Provider};
    ///
    /// # async fn example() -> any_llm_rs::Result<()> {
    /// let client = AnyLLM::create(Provider::OpenAI, None, None, None)?;
    /// let messages = vec![Message::user("Hello!")];
    /// let response = client.completion("gpt-4o", messages, None).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn completion(
        &self,
        model: &str,
        messages: Vec<Message>,
        options: Option<CompletionOptions>,
    ) -> Result<ChatCompletion> {
        match &self.provider {
            ProviderClient::OpenAI(provider) => provider.completion(model, messages, options).await,
        }
    }

    /// Perform a streaming chat completion request.
    ///
    /// # Arguments
    ///
    /// * `model` - Model ID to use for generation.
    /// * `messages` - The conversation messages.
    /// * `options` - Optional completion parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use any_llm_rs::{AnyLLM, Message, Provider};
    /// use futures::StreamExt;
    ///
    /// # async fn example() -> any_llm_rs::Result<()> {
    /// let client = AnyLLM::create(Provider::OpenAI, None, None, None)?;
    /// let messages = vec![Message::user("Tell me a story")];
    /// let mut stream = client.completion_stream("gpt-4o", messages, None).await?;
    ///
    /// while let Some(chunk) = stream.next().await {
    ///     let chunk = chunk?;
    ///     if let Some(delta) = chunk.choices.first().and_then(|c| c.delta.content.as_ref()) {
    ///         print!("{}", delta);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn completion_stream(
        &self,
        model: &str,
        messages: Vec<Message>,
        options: Option<CompletionOptions>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>> {
        match &self.provider {
            ProviderClient::OpenAI(provider) => {
                provider.completion_stream(model, messages, options).await
            }
        }
    }
}
