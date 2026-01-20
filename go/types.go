package anyllm

import (
	"encoding/json"
)

// Provider represents a supported LLM provider.
type Provider string

// Supported providers.
const (
	ProviderOpenAI     Provider = "openai"
	ProviderAnthropic  Provider = "anthropic"
	ProviderMistral    Provider = "mistral"
	ProviderGemini     Provider = "gemini"
	ProviderVertexAI   Provider = "vertexai"
	ProviderCohere     Provider = "cohere"
	ProviderCerebras   Provider = "cerebras"
	ProviderGroq       Provider = "groq"
	ProviderBedrock    Provider = "bedrock"
	ProviderAzure      Provider = "azure"
	ProviderAzureAI    Provider = "azureopenai"
	ProviderWatsonX    Provider = "watsonx"
	ProviderTogether   Provider = "together"
	ProviderSambanova  Provider = "sambanova"
	ProviderOllama     Provider = "ollama"
	ProviderMoonshot   Provider = "moonshot"
	ProviderNebius     Provider = "nebius"
	ProviderXAI        Provider = "xai"
	ProviderDatabricks Provider = "databricks"
	ProviderDeepSeek   Provider = "deepseek"
	ProviderInception  Provider = "inception"
	ProviderOpenRouter Provider = "openrouter"
	ProviderPortkey    Provider = "portkey"
	ProviderLMStudio   Provider = "lmstudio"
	ProviderLlama      Provider = "llama"
	ProviderVoyage     Provider = "voyage"
	ProviderPerplexity Provider = "perplexity"
	ProviderFireworks  Provider = "fireworks"
	ProviderHuggingFce Provider = "huggingface"
	ProviderLlamafile  Provider = "llamafile"
	ProviderLlamaCpp   Provider = "llamacpp"
	ProviderSageMaker  Provider = "sagemaker"
	ProviderMiniMax    Provider = "minimax"
	ProviderVLLM       Provider = "vllm"
	ProviderZAI        Provider = "zai"
)

// Message represents a chat message with role and content.
// Content can be a string or an array of content parts for multi-modal messages.
type Message struct {
	Role       string      `json:"role"`
	Content    interface{} `json:"content,omitempty"`
	Name       *string     `json:"name,omitempty"`
	ToolCalls  []ToolCall  `json:"tool_calls,omitempty"`
	ToolCallID *string     `json:"tool_call_id,omitempty"`
}

// ContentPart represents a content part in a multi-modal message.
type ContentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

// ImageURL represents an image URL in a content part.
type ImageURL struct {
	URL    string  `json:"url"`
	Detail *string `json:"detail,omitempty"`
}

// ToolCall represents a tool call made by the assistant.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall represents a function call within a tool call.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// Tool represents a tool that the model can call.
type Tool struct {
	Type     string             `json:"type"`
	Function FunctionDefinition `json:"function"`
}

// FunctionDefinition represents the definition of a function tool.
type FunctionDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
	Strict      *bool                  `json:"strict,omitempty"`
}

// ResponseFormat specifies the format of the model response.
type ResponseFormat struct {
	Type       string            `json:"type"`
	JSONSchema *JSONSchemaFormat `json:"json_schema,omitempty"`
}

// JSONSchemaFormat specifies a JSON schema for structured output.
type JSONSchemaFormat struct {
	Name        string                 `json:"name"`
	Schema      map[string]interface{} `json:"schema,omitempty"`
	Description string                 `json:"description,omitempty"`
	Strict      *bool                  `json:"strict,omitempty"`
}

// StreamOptions contains options for streaming responses.
type StreamOptions struct {
	IncludeUsage *bool `json:"include_usage,omitempty"`
}

// CompletionOptions contains optional parameters for chat completion requests.
type CompletionOptions struct {
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	MaxTokens           *int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	N                   *int            `json:"n,omitempty"`
	Stop                interface{}     `json:"stop,omitempty"` // string or []string
	PresencePenalty     *float64        `json:"presence_penalty,omitempty"`
	FrequencyPenalty    *float64        `json:"frequency_penalty,omitempty"`
	LogitBias           map[string]int  `json:"logit_bias,omitempty"`
	User                *string         `json:"user,omitempty"`
	Tools               []Tool          `json:"tools,omitempty"`
	ToolChoice          interface{}     `json:"tool_choice,omitempty"` // string or object
	ParallelToolCalls   *bool           `json:"parallel_tool_calls,omitempty"`
	ResponseFormat      *ResponseFormat `json:"response_format,omitempty"`
	Logprobs            *bool           `json:"logprobs,omitempty"`
	TopLogprobs         *int            `json:"top_logprobs,omitempty"`
	Seed                *int            `json:"seed,omitempty"`
	StreamOptions       *StreamOptions  `json:"stream_options,omitempty"`
}

// ChatCompletion represents a chat completion response.
type ChatCompletion struct {
	ID                string            `json:"id"`
	Object            string            `json:"object"`
	Created           int64             `json:"created"`
	Model             string            `json:"model"`
	Choices           []Choice          `json:"choices"`
	Usage             *Usage            `json:"usage,omitempty"`
	ServiceTier       string            `json:"service_tier,omitempty"`
	SystemFingerprint string            `json:"system_fingerprint,omitempty"`
	Validation        *ValidationResult `json:"_validation,omitempty"`
	RawResponse       *json.RawMessage  `json:"-"`
}

// Choice represents a completion choice.
type Choice struct {
	Index        int             `json:"index"`
	Message      ResponseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
	Logprobs     *Logprobs       `json:"logprobs,omitempty"`
}

// ResponseMessage represents the assistant's response message.
type ResponseMessage struct {
	Role      string     `json:"role"`
	Content   *string    `json:"content"`
	Refusal   *string    `json:"refusal,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// Logprobs represents log probability information.
type Logprobs struct {
	Content []TokenLogprob `json:"content,omitempty"`
}

// TokenLogprob represents log probability for a token.
type TokenLogprob struct {
	Token       string       `json:"token"`
	Logprob     float64      `json:"logprob"`
	Bytes       []int        `json:"bytes,omitempty"`
	TopLogprobs []TopLogprob `json:"top_logprobs,omitempty"`
}

// TopLogprob represents a top log probability entry.
type TopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes,omitempty"`
}

// Usage represents token usage statistics.
type Usage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

// PromptTokensDetails contains breakdown of prompt tokens.
type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
	AudioTokens  int `json:"audio_tokens,omitempty"`
}

// CompletionTokensDetails contains breakdown of completion tokens.
type CompletionTokensDetails struct {
	ReasoningTokens          int `json:"reasoning_tokens,omitempty"`
	AudioTokens              int `json:"audio_tokens,omitempty"`
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens,omitempty"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens,omitempty"`
}

// ValidationResult contains validation information from the acceptance test server.
type ValidationResult struct {
	Passed   bool              `json:"passed"`
	Errors   []ValidationError `json:"errors,omitempty"`
	Scenario string            `json:"scenario,omitempty"`
}

// ValidationError represents a validation error.
type ValidationError struct {
	Field   string `json:"field"`
	Message string `json:"message"`
}

// ChatCompletionChunk represents a streaming chunk of a chat completion.
type ChatCompletionChunk struct {
	ID                string        `json:"id"`
	Object            string        `json:"object"`
	Created           int64         `json:"created"`
	Model             string        `json:"model"`
	Choices           []ChunkChoice `json:"choices"`
	Usage             *Usage        `json:"usage,omitempty"`
	ServiceTier       string        `json:"service_tier,omitempty"`
	SystemFingerprint string        `json:"system_fingerprint,omitempty"`
}

// ChunkChoice represents a choice in a streaming chunk.
type ChunkChoice struct {
	Index        int         `json:"index"`
	Delta        ChoiceDelta `json:"delta"`
	FinishReason *string     `json:"finish_reason,omitempty"`
	Logprobs     *Logprobs   `json:"logprobs,omitempty"`
}

// ChoiceDelta represents the delta content in a streaming chunk.
type ChoiceDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   *string    `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	Refusal   *string    `json:"refusal,omitempty"`
}

// ClientConfig contains configuration for creating an LLM client.
type ClientConfig struct {
	APIKey         string
	APIBase        string
	DefaultHeaders map[string]string
}
