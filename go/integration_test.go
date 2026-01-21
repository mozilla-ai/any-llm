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

// TestFunctionToTool tests the FunctionToTool helper function.
func TestFunctionToTool(t *testing.T) {
	t.Run("basic_function", func(t *testing.T) {
		spec := anyllm.FunctionSpec{
			Name:        "get_weather",
			Description: "Get weather information for a location.",
			Parameters: []anyllm.ParameterSpec{
				{
					Name:        "location",
					Type:        reflect.TypeOf(""),
					Description: "The city name to get weather for.",
					Required:    true,
				},
				{
					Name:     "unit",
					Type:     reflect.TypeOf(""),
					Required: false,
					Enum:     []interface{}{"celsius", "fahrenheit"},
				},
			},
		}

		tool := anyllm.FunctionToTool(spec)

		assert.Equal(t, "function", tool.Type)
		assert.Equal(t, "get_weather", tool.Function.Name)
		assert.Equal(t, "Get weather information for a location.", tool.Function.Description)

		params := tool.Function.Parameters
		assert.NotNil(t, params)
		assert.Equal(t, "object", params["type"])

		properties := params["properties"].(map[string]interface{})
		assert.Contains(t, properties, "location")
		assert.Contains(t, properties, "unit")

		locationSchema := properties["location"].(map[string]interface{})
		assert.Equal(t, "string", locationSchema["type"])

		unitSchema := properties["unit"].(map[string]interface{})
		assert.Equal(t, "string", unitSchema["type"])
		assert.Equal(t, []interface{}{"celsius", "fahrenheit"}, unitSchema["enum"])

		required := params["required"].([]string)
		assert.Contains(t, required, "location")
		assert.NotContains(t, required, "unit")
	})

	t.Run("various_types", func(t *testing.T) {
		spec := anyllm.FunctionSpec{
			Name:        "test_func",
			Description: "A test function with various types.",
			Parameters: []anyllm.ParameterSpec{
				{Name: "str_param", Type: reflect.TypeOf(""), Required: true},
				{Name: "int_param", Type: reflect.TypeOf(0), Required: true},
				{Name: "float_param", Type: reflect.TypeOf(0.0), Required: true},
				{Name: "bool_param", Type: reflect.TypeOf(false), Required: true},
				{Name: "slice_param", Type: reflect.TypeOf([]string{}), Required: false},
				{Name: "map_param", Type: reflect.TypeOf(map[string]int{}), Required: false},
			},
		}

		tool := anyllm.FunctionToTool(spec)
		params := tool.Function.Parameters
		properties := params["properties"].(map[string]interface{})

		strSchema := properties["str_param"].(map[string]interface{})
		assert.Equal(t, "string", strSchema["type"])

		intSchema := properties["int_param"].(map[string]interface{})
		assert.Equal(t, "integer", intSchema["type"])

		floatSchema := properties["float_param"].(map[string]interface{})
		assert.Equal(t, "number", floatSchema["type"])

		boolSchema := properties["bool_param"].(map[string]interface{})
		assert.Equal(t, "boolean", boolSchema["type"])

		sliceSchema := properties["slice_param"].(map[string]interface{})
		assert.Equal(t, "array", sliceSchema["type"])
		assert.Equal(t, map[string]interface{}{"type": "string"}, sliceSchema["items"])

		mapSchema := properties["map_param"].(map[string]interface{})
		assert.Equal(t, "object", mapSchema["type"])
		assert.Equal(t, map[string]interface{}{"type": "integer"}, mapSchema["additionalProperties"])
	})

	t.Run("struct_with_json_skip_tag", func(t *testing.T) {
		// Test struct with json:"-" tag should be skipped
		type TestStruct struct {
			Name     string `json:"name"`
			Internal string `json:"-"`
			Count    int    `json:"count,omitempty"`
		}

		spec := anyllm.FunctionSpec{
			Name:        "test_struct_func",
			Description: "A test function with struct parameter.",
			Parameters: []anyllm.ParameterSpec{
				{Name: "data", Type: reflect.TypeOf(TestStruct{}), Required: true},
			},
		}

		tool := anyllm.FunctionToTool(spec)
		params := tool.Function.Parameters
		properties := params["properties"].(map[string]interface{})

		dataSchema := properties["data"].(map[string]interface{})
		assert.Equal(t, "object", dataSchema["type"])

		dataProperties := dataSchema["properties"].(map[string]interface{})
		// Should contain "name" and "count" but NOT "Internal" (json:"-")
		assert.Contains(t, dataProperties, "name")
		assert.Contains(t, dataProperties, "count")
		assert.NotContains(t, dataProperties, "Internal")
		assert.NotContains(t, dataProperties, "-")

		// "name" is required (no omitempty), "count" is optional (has omitempty)
		dataRequired := dataSchema["required"].([]string)
		assert.Contains(t, dataRequired, "name")
		assert.NotContains(t, dataRequired, "count")
	})
}
