package anyllm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/shared"
)

const (
	openAIProviderName = "OpenAI"
	openAIEnvVarName   = "OPENAI_API_KEY"
)

// OpenAIProvider implements the LLMClient interface for OpenAI using the official SDK.
type OpenAIProvider struct {
	client openai.Client
}

// NewOpenAIProvider creates a new OpenAI provider using the official openai-go SDK.
func NewOpenAIProvider(config *ClientConfig) (LLMClient, error) {
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv(openAIEnvVarName)
	}
	if apiKey == "" {
		return nil, NewMissingAPIKeyError(openAIProviderName, openAIEnvVarName)
	}

	opts := []option.RequestOption{
		option.WithAPIKey(apiKey),
	}

	if config.APIBase != "" {
		opts = append(opts, option.WithBaseURL(config.APIBase))
	}

	for key, value := range config.DefaultHeaders {
		opts = append(opts, option.WithHeader(key, value))
	}

	client := openai.NewClient(opts...)

	return &OpenAIProvider{
		client: client,
	}, nil
}

// ProviderName returns the name of the provider.
func (p *OpenAIProvider) ProviderName() string {
	return openAIProviderName
}

// Completion performs a chat completion request using the OpenAI SDK.
func (p *OpenAIProvider) Completion(ctx context.Context, model string, messages []Message, opts *CompletionOptions) (*ChatCompletion, error) {
	params := p.buildCompletionParams(model, messages, opts)

	resp, err := p.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, p.mapSDKError(err)
	}

	return p.convertResponse(resp), nil
}

// CompletionStream performs a streaming chat completion request using the OpenAI SDK.
func (p *OpenAIProvider) CompletionStream(ctx context.Context, model string, messages []Message, opts *CompletionOptions) (<-chan ChatCompletionChunk, <-chan error) {
	chunkChan := make(chan ChatCompletionChunk)
	errChan := make(chan error, 1)

	go func() {
		defer close(chunkChan)
		defer close(errChan)

		params := p.buildCompletionParams(model, messages, opts)

		stream := p.client.Chat.Completions.NewStreaming(ctx, params)

		for stream.Next() {
			chunk := stream.Current()
			converted := p.convertChunk(chunk)

			select {
			case chunkChan <- converted:
			case <-ctx.Done():
				return
			}
		}

		if err := stream.Err(); err != nil {
			errChan <- p.mapSDKError(err)
		}
	}()

	return chunkChan, errChan
}

// buildCompletionParams builds the OpenAI SDK parameters from our types.
func (p *OpenAIProvider) buildCompletionParams(model string, messages []Message, opts *CompletionOptions) openai.ChatCompletionNewParams {
	params := openai.ChatCompletionNewParams{
		Model:    shared.ChatModel(model),
		Messages: p.convertMessages(messages),
	}

	if opts == nil {
		return params
	}

	if opts.Temperature != nil {
		params.Temperature = openai.Float(*opts.Temperature)
	}
	if opts.TopP != nil {
		params.TopP = openai.Float(*opts.TopP)
	}
	if opts.MaxTokens != nil {
		params.MaxTokens = openai.Int(int64(*opts.MaxTokens))
	}
	if opts.MaxCompletionTokens != nil {
		params.MaxCompletionTokens = openai.Int(int64(*opts.MaxCompletionTokens))
	}
	if opts.N != nil {
		params.N = openai.Int(int64(*opts.N))
	}
	if opts.PresencePenalty != nil {
		params.PresencePenalty = openai.Float(*opts.PresencePenalty)
	}
	if opts.FrequencyPenalty != nil {
		params.FrequencyPenalty = openai.Float(*opts.FrequencyPenalty)
	}
	if opts.User != nil {
		params.User = openai.String(*opts.User)
	}
	if opts.Logprobs != nil {
		params.Logprobs = openai.Bool(*opts.Logprobs)
	}
	if opts.TopLogprobs != nil {
		params.TopLogprobs = openai.Int(int64(*opts.TopLogprobs))
	}
	if opts.Seed != nil {
		params.Seed = openai.Int(int64(*opts.Seed))
	}
	if opts.Stop != nil {
		params.Stop = p.convertStop(opts.Stop)
	}
	if opts.Tools != nil && len(opts.Tools) > 0 {
		params.Tools = p.convertTools(opts.Tools)
	}
	if opts.ResponseFormat != nil {
		params.ResponseFormat = p.convertResponseFormat(opts.ResponseFormat)
	}
	if opts.StreamOptions != nil {
		params.StreamOptions = p.convertStreamOptions(opts.StreamOptions)
	}
	if opts.ParallelToolCalls != nil {
		params.ParallelToolCalls = openai.Bool(*opts.ParallelToolCalls)
	}

	return params
}

// convertMessages converts our Message types to OpenAI SDK message types.
func (p *OpenAIProvider) convertMessages(messages []Message) []openai.ChatCompletionMessageParamUnion {
	result := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			content := p.getStringContent(msg.Content)
			result = append(result, openai.SystemMessage(content))

		case "user":
			if parts, ok := msg.Content.([]interface{}); ok {
				// Multi-modal content
				contentParts := p.convertContentParts(parts)
				result = append(result, openai.ChatCompletionMessageParamUnion{
					OfUser: &openai.ChatCompletionUserMessageParam{
						Content: openai.ChatCompletionUserMessageParamContentUnion{
							OfArrayOfContentParts: contentParts,
						},
					},
				})
			} else {
				// Simple string content
				content := p.getStringContent(msg.Content)
				result = append(result, openai.UserMessage(content))
			}

		case "assistant":
			assistantMsg := &openai.ChatCompletionAssistantMessageParam{}
			if msg.Content != nil {
				content := p.getStringContent(msg.Content)
				assistantMsg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{
					OfString: openai.String(content),
				}
			}
			if len(msg.ToolCalls) > 0 {
				toolCalls := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(msg.ToolCalls))
				for _, tc := range msg.ToolCalls {
					toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
						OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
							ID: tc.ID,
							Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
								Name:      tc.Function.Name,
								Arguments: tc.Function.Arguments,
							},
						},
					})
				}
				assistantMsg.ToolCalls = toolCalls
			}
			result = append(result, openai.ChatCompletionMessageParamUnion{
				OfAssistant: assistantMsg,
			})

		case "tool":
			toolCallID := ""
			if msg.ToolCallID != nil {
				toolCallID = *msg.ToolCallID
			}
			content := p.getStringContent(msg.Content)
			result = append(result, openai.ToolMessage(content, toolCallID))
		}
	}

	return result
}

// convertContentParts converts multi-modal content parts.
func (p *OpenAIProvider) convertContentParts(parts []interface{}) []openai.ChatCompletionContentPartUnionParam {
	result := make([]openai.ChatCompletionContentPartUnionParam, 0, len(parts))

	for _, part := range parts {
		partMap, ok := part.(map[string]interface{})
		if !ok {
			continue
		}

		partType, _ := partMap["type"].(string)

		switch partType {
		case "text":
			text, _ := partMap["text"].(string)
			result = append(result, openai.ChatCompletionContentPartUnionParam{
				OfText: &openai.ChatCompletionContentPartTextParam{
					Text: text,
				},
			})

		case "image_url":
			if imgURL, ok := partMap["image_url"].(map[string]interface{}); ok {
				url, _ := imgURL["url"].(string)
				imgParam := &openai.ChatCompletionContentPartImageParam{
					ImageURL: openai.ChatCompletionContentPartImageImageURLParam{
						URL: url,
					},
				}
				if detail, ok := imgURL["detail"].(string); ok {
					imgParam.ImageURL.Detail = detail
				}
				result = append(result, openai.ChatCompletionContentPartUnionParam{
					OfImageURL: imgParam,
				})
			}
		}
	}

	return result
}

// getStringContent extracts string content from interface{}.
func (p *OpenAIProvider) getStringContent(content interface{}) string {
	if content == nil {
		return ""
	}
	if str, ok := content.(string); ok {
		return str
	}
	// For complex content, marshal to JSON
	if b, err := json.Marshal(content); err == nil {
		return string(b)
	}
	return fmt.Sprintf("%v", content)
}

// convertStop converts stop sequences.
func (p *OpenAIProvider) convertStop(stop interface{}) openai.ChatCompletionNewParamsStopUnion {
	switch v := stop.(type) {
	case string:
		return openai.ChatCompletionNewParamsStopUnion{
			OfString: openai.String(v),
		}
	case []string:
		return openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: v,
		}
	case []interface{}:
		strs := make([]string, 0, len(v))
		for _, s := range v {
			if str, ok := s.(string); ok {
				strs = append(strs, str)
			}
		}
		return openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: strs,
		}
	}
	return openai.ChatCompletionNewParamsStopUnion{}
}

// convertTools converts our Tool types to OpenAI SDK tool types.
func (p *OpenAIProvider) convertTools(tools []Tool) []openai.ChatCompletionToolUnionParam {
	result := make([]openai.ChatCompletionToolUnionParam, 0, len(tools))

	for _, tool := range tools {
		funcDef := shared.FunctionDefinitionParam{
			Name: tool.Function.Name,
		}
		if tool.Function.Description != "" {
			funcDef.Description = openai.String(tool.Function.Description)
		}
		if tool.Function.Parameters != nil {
			funcDef.Parameters = shared.FunctionParameters(tool.Function.Parameters)
		}
		if tool.Function.Strict != nil {
			funcDef.Strict = openai.Bool(*tool.Function.Strict)
		}

		result = append(result, openai.ChatCompletionToolUnionParam{
			OfFunction: &openai.ChatCompletionFunctionToolParam{
				Function: funcDef,
			},
		})
	}

	return result
}

// convertResponseFormat converts our ResponseFormat to OpenAI SDK format.
func (p *OpenAIProvider) convertResponseFormat(rf *ResponseFormat) openai.ChatCompletionNewParamsResponseFormatUnion {
	switch rf.Type {
	case "text":
		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfText: &shared.ResponseFormatTextParam{},
		}
	case "json_object":
		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &shared.ResponseFormatJSONObjectParam{},
		}
	case "json_schema":
		if rf.JSONSchema != nil {
			schemaParam := shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name: rf.JSONSchema.Name,
			}
			if rf.JSONSchema.Description != "" {
				schemaParam.Description = openai.String(rf.JSONSchema.Description)
			}
			if rf.JSONSchema.Schema != nil {
				schemaParam.Schema = rf.JSONSchema.Schema
			}
			if rf.JSONSchema.Strict != nil {
				schemaParam.Strict = openai.Bool(*rf.JSONSchema.Strict)
			}
			return openai.ChatCompletionNewParamsResponseFormatUnion{
				OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
					JSONSchema: schemaParam,
				},
			}
		}
	}
	return openai.ChatCompletionNewParamsResponseFormatUnion{}
}

// convertStreamOptions converts our StreamOptions to OpenAI SDK format.
func (p *OpenAIProvider) convertStreamOptions(so *StreamOptions) openai.ChatCompletionStreamOptionsParam {
	result := openai.ChatCompletionStreamOptionsParam{}
	if so.IncludeUsage != nil {
		result.IncludeUsage = openai.Bool(*so.IncludeUsage)
	}
	return result
}

// convertResponse converts the OpenAI SDK response to our ChatCompletion type.
func (p *OpenAIProvider) convertResponse(resp *openai.ChatCompletion) *ChatCompletion {
	result := &ChatCompletion{
		ID:                resp.ID,
		Object:            string(resp.Object),
		Created:           resp.Created,
		Model:             resp.Model,
		ServiceTier:       string(resp.ServiceTier),
		SystemFingerprint: resp.SystemFingerprint,
	}

	// Convert choices
	result.Choices = make([]Choice, len(resp.Choices))
	for i, choice := range resp.Choices {
		content := choice.Message.Content
		refusal := choice.Message.Refusal
		result.Choices[i] = Choice{
			Index:        int(choice.Index),
			FinishReason: string(choice.FinishReason),
			Message: ResponseMessage{
				Role:    string(choice.Message.Role),
				Content: &content,
				Refusal: &refusal,
			},
		}

		// Convert tool calls if present
		if len(choice.Message.ToolCalls) > 0 {
			result.Choices[i].Message.ToolCalls = make([]ToolCall, len(choice.Message.ToolCalls))
			for j, tc := range choice.Message.ToolCalls {
				// Use direct fields from the union type
				result.Choices[i].Message.ToolCalls[j] = ToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}
	}

	// Convert usage if present
	if resp.Usage.TotalTokens > 0 {
		result.Usage = &Usage{
			PromptTokens:     int(resp.Usage.PromptTokens),
			CompletionTokens: int(resp.Usage.CompletionTokens),
			TotalTokens:      int(resp.Usage.TotalTokens),
		}
	}

	// Parse validation result from raw JSON if present
	result.Validation = p.extractValidation(resp)

	return result
}

// extractValidation extracts the _validation field from the raw response.
func (p *OpenAIProvider) extractValidation(resp *openai.ChatCompletion) *ValidationResult {
	// The openai-go SDK provides raw JSON access through the RawJSON method
	rawJSON := resp.RawJSON()
	if len(rawJSON) == 0 {
		return nil
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(rawJSON), &raw); err != nil {
		return nil
	}

	validationRaw, ok := raw["_validation"]
	if !ok {
		return nil
	}

	var validation ValidationResult
	if err := json.Unmarshal(validationRaw, &validation); err != nil {
		return nil
	}

	return &validation
}

// convertChunk converts the OpenAI SDK chunk to our ChatCompletionChunk type.
func (p *OpenAIProvider) convertChunk(chunk openai.ChatCompletionChunk) ChatCompletionChunk {
	result := ChatCompletionChunk{
		ID:                chunk.ID,
		Object:            string(chunk.Object),
		Created:           chunk.Created,
		Model:             chunk.Model,
		ServiceTier:       string(chunk.ServiceTier),
		SystemFingerprint: chunk.SystemFingerprint,
	}

	// Convert choices
	result.Choices = make([]ChunkChoice, len(chunk.Choices))
	for i, choice := range chunk.Choices {
		var finishReason *string
		if choice.FinishReason != "" {
			fr := string(choice.FinishReason)
			finishReason = &fr
		}

		content := choice.Delta.Content
		refusal := choice.Delta.Refusal

		result.Choices[i] = ChunkChoice{
			Index:        int(choice.Index),
			FinishReason: finishReason,
			Delta: ChoiceDelta{
				Role:    string(choice.Delta.Role),
				Content: &content,
				Refusal: &refusal,
			},
		}

		// Convert tool calls if present
		if len(choice.Delta.ToolCalls) > 0 {
			result.Choices[i].Delta.ToolCalls = make([]ToolCall, len(choice.Delta.ToolCalls))
			for j, tc := range choice.Delta.ToolCalls {
				result.Choices[i].Delta.ToolCalls[j] = ToolCall{
					ID:   tc.ID,
					Type: string(tc.Type),
					Function: FunctionCall{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}
	}

	// Convert usage if present
	if chunk.Usage.TotalTokens > 0 {
		result.Usage = &Usage{
			PromptTokens:     int(chunk.Usage.PromptTokens),
			CompletionTokens: int(chunk.Usage.CompletionTokens),
			TotalTokens:      int(chunk.Usage.TotalTokens),
		}
	}

	return result
}

// mapSDKError maps OpenAI SDK errors to our error types.
func (p *OpenAIProvider) mapSDKError(err error) error {
	if err == nil {
		return nil
	}

	message := err.Error()
	lowerMessage := strings.ToLower(message)

	// Check for specific error patterns in the message
	if strings.Contains(lowerMessage, "authentication") || strings.Contains(lowerMessage, "401") || strings.Contains(lowerMessage, "invalid api key") {
		return NewAuthenticationError(message, err, openAIProviderName)
	}

	if strings.Contains(lowerMessage, "rate limit") || strings.Contains(lowerMessage, "429") {
		return NewRateLimitError(message, err, openAIProviderName)
	}

	if strings.Contains(lowerMessage, "context length") || strings.Contains(lowerMessage, "maximum context") {
		return NewContextLengthExceededError(message, err, openAIProviderName)
	}

	if strings.Contains(lowerMessage, "content filter") || strings.Contains(lowerMessage, "content_filter") {
		return NewContentFilterError(message, err, openAIProviderName)
	}

	if strings.Contains(lowerMessage, "model") && strings.Contains(lowerMessage, "not found") {
		return NewModelNotFoundError(message, err, openAIProviderName)
	}

	if strings.Contains(lowerMessage, "invalid") || strings.Contains(lowerMessage, "400") {
		return NewInvalidRequestError(message, err, openAIProviderName)
	}

	if strings.Contains(lowerMessage, "500") || strings.Contains(lowerMessage, "internal server") {
		return NewProviderError(message, err, openAIProviderName)
	}

	// Default to provider error
	return NewProviderError(message, err, openAIProviderName)
}
