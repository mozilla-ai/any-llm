package anyllm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

const (
	openAIProviderName = "OpenAI"
	openAIEnvVarName   = "OPENAI_API_KEY"
	openAIDefaultBase  = "https://api.openai.com/v1"
)

// OpenAIProvider implements the LLMClient interface for OpenAI.
type OpenAIProvider struct {
	apiKey         string
	apiBase        string
	httpClient     *http.Client
	defaultHeaders map[string]string
}

// NewOpenAIProvider creates a new OpenAI provider.
func NewOpenAIProvider(config *ClientConfig) (LLMClient, error) {
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv(openAIEnvVarName)
	}
	if apiKey == "" {
		return nil, NewMissingAPIKeyError(openAIProviderName, openAIEnvVarName)
	}

	apiBase := config.APIBase
	if apiBase == "" {
		apiBase = openAIDefaultBase
	}

	return &OpenAIProvider{
		apiKey:         apiKey,
		apiBase:        apiBase,
		httpClient:     &http.Client{},
		defaultHeaders: config.DefaultHeaders,
	}, nil
}

// ProviderName returns the name of the provider.
func (p *OpenAIProvider) ProviderName() string {
	return openAIProviderName
}

// Completion performs a chat completion request.
func (p *OpenAIProvider) Completion(ctx context.Context, model string, messages []Message, opts *CompletionOptions) (*ChatCompletion, error) {
	params := p.buildCompletionParams(model, messages, opts, false)

	body, err := json.Marshal(params)
	if err != nil {
		return nil, NewInvalidRequestError(fmt.Sprintf("failed to marshal request: %v", err), err, openAIProviderName)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.apiBase+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, NewProviderError(fmt.Sprintf("failed to create request: %v", err), err, openAIProviderName)
	}

	p.setHeaders(req)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, NewProviderError(fmt.Sprintf("request failed: %v", err), err, openAIProviderName)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, NewProviderError(fmt.Sprintf("failed to read response: %v", err), err, openAIProviderName)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, p.mapHTTPError(resp.StatusCode, respBody)
	}

	var completion ChatCompletion
	if err := json.Unmarshal(respBody, &completion); err != nil {
		return nil, NewProviderError(fmt.Sprintf("failed to parse response: %v", err), err, openAIProviderName)
	}

	return &completion, nil
}

// CompletionStream performs a streaming chat completion request.
func (p *OpenAIProvider) CompletionStream(ctx context.Context, model string, messages []Message, opts *CompletionOptions) (<-chan ChatCompletionChunk, <-chan error) {
	chunkChan := make(chan ChatCompletionChunk)
	errChan := make(chan error, 1)

	go func() {
		defer close(chunkChan)
		defer close(errChan)

		params := p.buildCompletionParams(model, messages, opts, true)

		body, err := json.Marshal(params)
		if err != nil {
			errChan <- NewInvalidRequestError(fmt.Sprintf("failed to marshal request: %v", err), err, openAIProviderName)
			return
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.apiBase+"/chat/completions", bytes.NewReader(body))
		if err != nil {
			errChan <- NewProviderError(fmt.Sprintf("failed to create request: %v", err), err, openAIProviderName)
			return
		}

		p.setHeaders(req)

		resp, err := p.httpClient.Do(req)
		if err != nil {
			errChan <- NewProviderError(fmt.Sprintf("request failed: %v", err), err, openAIProviderName)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			respBody, _ := io.ReadAll(resp.Body)
			errChan <- p.mapHTTPError(resp.StatusCode, respBody)
			return
		}

		reader := bufio.NewReader(resp.Body)
		for {
			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err == io.EOF {
					break
				}
				errChan <- NewProviderError(fmt.Sprintf("failed to read stream: %v", err), err, openAIProviderName)
				return
			}

			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}

			// SSE format: "data: {json}"
			if !bytes.HasPrefix(line, []byte("data: ")) {
				continue
			}

			data := bytes.TrimPrefix(line, []byte("data: "))
			if string(data) == "[DONE]" {
				break
			}

			var chunk ChatCompletionChunk
			if err := json.Unmarshal(data, &chunk); err != nil {
				// Skip malformed chunks
				continue
			}

			select {
			case chunkChan <- chunk:
			case <-ctx.Done():
				return
			}
		}
	}()

	return chunkChan, errChan
}

// setHeaders sets the required headers for OpenAI API requests.
func (p *OpenAIProvider) setHeaders(req *http.Request) {
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	for key, value := range p.defaultHeaders {
		req.Header.Set(key, value)
	}
}

// buildCompletionParams builds the request parameters for a completion.
func (p *OpenAIProvider) buildCompletionParams(model string, messages []Message, opts *CompletionOptions, stream bool) map[string]interface{} {
	params := map[string]interface{}{
		"model":    model,
		"messages": messages,
	}

	if stream {
		params["stream"] = true
	}

	if opts == nil {
		return params
	}

	if opts.Temperature != nil {
		params["temperature"] = *opts.Temperature
	}
	if opts.TopP != nil {
		params["top_p"] = *opts.TopP
	}
	if opts.MaxTokens != nil {
		params["max_tokens"] = *opts.MaxTokens
	}
	if opts.MaxCompletionTokens != nil {
		params["max_completion_tokens"] = *opts.MaxCompletionTokens
	}
	if opts.N != nil {
		params["n"] = *opts.N
	}
	if opts.Stop != nil {
		params["stop"] = opts.Stop
	}
	if opts.PresencePenalty != nil {
		params["presence_penalty"] = *opts.PresencePenalty
	}
	if opts.FrequencyPenalty != nil {
		params["frequency_penalty"] = *opts.FrequencyPenalty
	}
	if opts.LogitBias != nil {
		params["logit_bias"] = opts.LogitBias
	}
	if opts.User != nil {
		params["user"] = *opts.User
	}
	if opts.Tools != nil && len(opts.Tools) > 0 {
		params["tools"] = opts.Tools
	}
	if opts.ToolChoice != nil {
		params["tool_choice"] = opts.ToolChoice
	}
	if opts.ParallelToolCalls != nil {
		params["parallel_tool_calls"] = *opts.ParallelToolCalls
	}
	if opts.ResponseFormat != nil {
		params["response_format"] = opts.ResponseFormat
	}
	if opts.Logprobs != nil {
		params["logprobs"] = *opts.Logprobs
	}
	if opts.TopLogprobs != nil {
		params["top_logprobs"] = *opts.TopLogprobs
	}
	if opts.Seed != nil {
		params["seed"] = *opts.Seed
	}
	if opts.StreamOptions != nil {
		streamOpts := make(map[string]interface{})
		if opts.StreamOptions.IncludeUsage != nil {
			streamOpts["include_usage"] = *opts.StreamOptions.IncludeUsage
		}
		params["stream_options"] = streamOpts
	}

	return params
}

// mapHTTPError maps HTTP status codes to appropriate error types.
func (p *OpenAIProvider) mapHTTPError(statusCode int, body []byte) error {
	// Try to extract error message from response body
	var apiError struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Code    string `json:"code"`
		} `json:"error"`
	}

	message := string(body)
	if err := json.Unmarshal(body, &apiError); err == nil && apiError.Error.Message != "" {
		message = apiError.Error.Message
	}

	switch statusCode {
	case http.StatusUnauthorized:
		return NewAuthenticationError(message, nil, openAIProviderName)
	case http.StatusTooManyRequests:
		return NewRateLimitError(message, nil, openAIProviderName)
	case http.StatusBadRequest:
		lowerMessage := strings.ToLower(message)
		if strings.Contains(lowerMessage, "context length") || strings.Contains(lowerMessage, "maximum context") {
			return NewContextLengthExceededError(message, nil, openAIProviderName)
		}
		if strings.Contains(lowerMessage, "content filter") || strings.Contains(lowerMessage, "content_filter") {
			return NewContentFilterError(message, nil, openAIProviderName)
		}
		return NewInvalidRequestError(message, nil, openAIProviderName)
	case http.StatusNotFound:
		if strings.Contains(strings.ToLower(message), "model") {
			return NewModelNotFoundError(message, nil, openAIProviderName)
		}
		return NewInvalidRequestError(message, nil, openAIProviderName)
	default:
		if statusCode >= 500 {
			return NewProviderError(message, nil, openAIProviderName)
		}
		return NewProviderError(fmt.Sprintf("unexpected status code %d: %s", statusCode, message), nil, openAIProviderName)
	}
}
