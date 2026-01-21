package anyllm_test

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	anyllm "github.com/mozilla-ai/any-llm/go"
)

// getWeather is a mock function that returns weather data for a location.
func getWeather(location string) string {
	data := map[string]interface{}{
		"location":    location,
		"temperature": "15C",
		"condition":   "sunny",
	}
	result, _ := json.Marshal(data)
	return string(result)
}

// TestAgentLoopParallelToolCalls mirrors the Python integration test test_agent_loop_parallel_tool_calls.
// It tests the ability to make tool calls with an LLM and handle the tool results.
func TestAgentLoopParallelToolCalls(t *testing.T) {
	// Skip if no API key is available
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping integration test")
	}

	testCases := []struct {
		name         string
		provider     anyllm.Provider
		model        string
		clientConfig *anyllm.ClientConfig
	}{
		{
			name:     "OpenAI_gpt-5-nano",
			provider: anyllm.ProviderOpenAI,
			model:    "gpt-5-nano",
			clientConfig: &anyllm.ClientConfig{
				APIKey: apiKey,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create the LLM client
			llm, err := anyllm.Create(tc.provider, tc.clientConfig)
			require.NoError(t, err, "Failed to create LLM client")

			// Create the get_weather tool using FunctionToTool
			weatherTool := anyllm.FunctionToTool(anyllm.FunctionSpec{
				Name:        "get_weather",
				Description: "Get the weather for a location.",
				Parameters: []anyllm.ParameterSpec{
					{
						Name:        "location",
						Type:        reflect.TypeOf(""),
						Description: "The city name to get weather for.",
						Required:    true,
					},
				},
			})

			// Initial messages
			messages := []anyllm.Message{
				{
					Role:    "user",
					Content: "Get the weather for both Paris and London using the get_weather tool. Call the tool twice, once for each city.",
				},
			}

			// First completion - should return tool calls
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			result, err := llm.Completion(ctx, tc.model, messages, &anyllm.CompletionOptions{
				Tools: []anyllm.Tool{weatherTool},
			})
			require.NoError(t, err, "First completion failed")
			require.NotEmpty(t, result.Choices, "Response has no choices")

			toolCalls := result.Choices[0].Message.ToolCalls
			require.NotEmpty(t, toolCalls, "Expected tool calls, got: %v", result.Choices[0].Message)

			// Add assistant message with tool calls to conversation
			assistantContent := ""
			if result.Choices[0].Message.Content != nil {
				assistantContent = *result.Choices[0].Message.Content
			}
			messages = append(messages, anyllm.Message{
				Role:      "assistant",
				Content:   assistantContent,
				ToolCalls: toolCalls,
			})

			// Process each tool call
			for _, toolCall := range toolCalls {
				assert.Equal(t, "get_weather", toolCall.Function.Name, "Expected get_weather tool call")

				// Parse arguments and call the function
				var args struct {
					Location string `json:"location"`
				}
				err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args)
				require.NoError(t, err, "Failed to parse tool call arguments")

				// Execute the tool
				toolResult := getWeather(args.Location)

				// Add tool result to messages
				toolCallID := toolCall.ID
				toolName := toolCall.Function.Name
				messages = append(messages, anyllm.Message{
					Role:       "tool",
					Content:    toolResult,
					ToolCallID: &toolCallID,
					Name:       &toolName,
				})
			}

			// Second completion - should return final response
			secondResult, err := llm.Completion(ctx, tc.model, messages, &anyllm.CompletionOptions{
				Tools: []anyllm.Tool{weatherTool},
			})
			require.NoError(t, err, "Second completion failed")
			require.NotEmpty(t, secondResult.Choices, "Second response has no choices")

			// Either should have content or more tool calls
			hasContent := secondResult.Choices[0].Message.Content != nil && *secondResult.Choices[0].Message.Content != ""
			hasToolCalls := len(secondResult.Choices[0].Message.ToolCalls) > 0
			assert.True(t, hasContent || hasToolCalls, "Expected either content or tool calls in final response")
		})
	}
}
