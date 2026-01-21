//! `OpenAI` provider implementation.
//!
//! This module provides the `OpenAI` provider for the any-llm library.

use std::collections::HashMap;
use std::env;
use std::pin::Pin;

use async_stream::try_stream;
use futures::Stream;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};

use crate::errors::{AnyLLMError, Result};
use crate::types::{ChatCompletion, ChatCompletionChunk, CompletionOptions, Message};

/// Default `OpenAI` API base URL.
const DEFAULT_API_BASE: &str = "https://api.openai.com/v1";

/// Environment variable name for `OpenAI` API key.
const ENV_VAR_NAME: &str = "OPENAI_API_KEY";

/// Provider name.
const PROVIDER_NAME: &str = "OpenAI";

/// `OpenAI` completion request body.
#[derive(Debug, Serialize)]
struct CompletionRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logit_bias: Option<HashMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<serde_json::Value>,
}

/// `OpenAI` API error response.
#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    error: ApiError,
}

/// `OpenAI` API error details.
#[derive(Debug, Deserialize)]
struct ApiError {
    message: String,
    #[serde(rename = "type")]
    error_type: Option<String>,
    code: Option<String>,
}

/// `OpenAI` provider client.
#[derive(Debug)]
pub struct OpenAIProvider {
    client: Client,
    api_base: String,
    headers: HeaderMap,
}

impl OpenAIProvider {
    /// Create a new `OpenAI` provider.
    ///
    /// # Arguments
    ///
    /// * `api_key` - API key. If None, reads from `OPENAI_API_KEY` environment variable.
    /// * `api_base` - API base URL. If None, uses the default `OpenAI` URL.
    /// * `extra_headers` - Additional headers to include in requests.
    ///
    /// # Errors
    ///
    /// Returns an error if no API key is provided and the environment variable is not set.
    pub fn new(
        api_key: Option<String>,
        api_base: Option<String>,
        extra_headers: Option<HashMap<String, String>>,
    ) -> Result<Self> {
        let api_key = api_key
            .or_else(|| env::var(ENV_VAR_NAME).ok())
            .ok_or_else(|| AnyLLMError::missing_api_key(PROVIDER_NAME, ENV_VAR_NAME))?;

        let api_base = api_base.unwrap_or_else(|| DEFAULT_API_BASE.to_string());

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {api_key}"))
                .map_err(|e| AnyLLMError::invalid_request(PROVIDER_NAME, e.to_string()))?,
        );

        if let Some(extra) = extra_headers {
            for (key, value) in extra {
                let header_name = HeaderName::try_from(&key)
                    .map_err(|e| AnyLLMError::invalid_request(PROVIDER_NAME, e.to_string()))?;
                let header_value = HeaderValue::from_str(&value)
                    .map_err(|e| AnyLLMError::invalid_request(PROVIDER_NAME, e.to_string()))?;
                headers.insert(header_name, header_value);
            }
        }

        let client = Client::new();

        Ok(Self {
            client,
            api_base,
            headers,
        })
    }

    /// Convert completion options to request parameters.
    fn convert_completion_params(
        model: &str,
        messages: Vec<Message>,
        options: Option<CompletionOptions>,
        stream: bool,
    ) -> Result<CompletionRequest> {
        let options = options.unwrap_or_default();

        let tools = options.tools.map(serde_json::to_value).transpose()?;
        let tool_choice = options.tool_choice.map(serde_json::to_value).transpose()?;
        let response_format = options
            .response_format
            .map(serde_json::to_value)
            .transpose()?;
        let stream_options = if stream {
            options
                .stream_options
                .map(serde_json::to_value)
                .transpose()?
        } else {
            None
        };

        Ok(CompletionRequest {
            model: model.to_string(),
            messages,
            temperature: options.temperature,
            top_p: options.top_p,
            max_tokens: options.max_tokens,
            max_completion_tokens: options.max_completion_tokens,
            n: options.n,
            stop: options.stop,
            presence_penalty: options.presence_penalty,
            frequency_penalty: options.frequency_penalty,
            logit_bias: options.logit_bias,
            logprobs: options.logprobs,
            top_logprobs: options.top_logprobs,
            seed: options.seed,
            user: options.user,
            tools,
            tool_choice,
            parallel_tool_calls: options.parallel_tool_calls,
            response_format,
            stream: if stream { Some(true) } else { None },
            stream_options,
        })
    }

    /// Map API error to `AnyLLMError`.
    fn map_error(status: u16, error: ApiError) -> AnyLLMError {
        let code = error.code.as_deref().unwrap_or("");
        let error_type = error.error_type.as_deref().unwrap_or("");

        match status {
            401 => AnyLLMError::authentication(PROVIDER_NAME, Some(error.message)),
            429 => AnyLLMError::rate_limit(PROVIDER_NAME, Some(error.message)),
            400 => {
                if code == "context_length_exceeded" || error_type == "context_length_exceeded" {
                    AnyLLMError::context_length_exceeded(PROVIDER_NAME, Some(error.message))
                } else if code == "model_not_found" || error_type == "model_not_found" {
                    AnyLLMError::model_not_found(PROVIDER_NAME, Some(error.message))
                } else {
                    AnyLLMError::invalid_request(PROVIDER_NAME, error.message)
                }
            }
            403 => {
                if code == "content_policy_violation" || error_type == "content_filter" {
                    AnyLLMError::content_filter(PROVIDER_NAME, Some(error.message))
                } else {
                    AnyLLMError::authentication(PROVIDER_NAME, Some(error.message))
                }
            }
            404 => AnyLLMError::model_not_found(PROVIDER_NAME, Some(error.message)),
            _ => AnyLLMError::provider_error(PROVIDER_NAME, Some(error.message)),
        }
    }

    /// Perform a completion request.
    ///
    /// # Arguments
    ///
    /// * `model` - Model ID to use.
    /// * `messages` - Conversation messages.
    /// * `options` - Optional completion parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails or the response cannot be parsed.
    pub async fn completion(
        &self,
        model: &str,
        messages: Vec<Message>,
        options: Option<CompletionOptions>,
    ) -> Result<ChatCompletion> {
        let request_body = Self::convert_completion_params(model, messages, options, false)?;

        let url = format!("{}/chat/completions", self.api_base);
        let response = self
            .client
            .post(&url)
            .headers(self.headers.clone())
            .json(&request_body)
            .send()
            .await?;

        let status = response.status().as_u16();

        if !response.status().is_success() {
            let error_response: ApiErrorResponse = response.json().await?;
            return Err(Self::map_error(status, error_response.error));
        }

        let completion: ChatCompletion = response.json().await?;
        Ok(completion)
    }

    /// Perform a streaming completion request.
    ///
    /// # Arguments
    ///
    /// * `model` - Model ID to use.
    /// * `messages` - Conversation messages.
    /// * `options` - Optional completion parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the request fails.
    pub async fn completion_stream(
        &self,
        model: &str,
        messages: Vec<Message>,
        options: Option<CompletionOptions>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatCompletionChunk>> + Send>>> {
        let request_body = Self::convert_completion_params(model, messages, options, true)?;

        let url = format!("{}/chat/completions", self.api_base);
        let response = self
            .client
            .post(&url)
            .headers(self.headers.clone())
            .json(&request_body)
            .send()
            .await?;

        let status = response.status().as_u16();

        if !response.status().is_success() {
            let error_response: ApiErrorResponse = response.json().await?;
            return Err(Self::map_error(status, error_response.error));
        }

        let stream = try_stream! {
            use futures::StreamExt;

            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                let chunk = chunk_result?;
                let text = String::from_utf8_lossy(&chunk);
                buffer.push_str(&text);

                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].to_string();
                    buffer = buffer[pos + 1..].to_string();

                    let line = line.trim();
                    if line.is_empty() {
                        continue;
                    }

                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            return;
                        }

                        let chunk: ChatCompletionChunk = serde_json::from_str(data)?;
                        yield chunk;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}
