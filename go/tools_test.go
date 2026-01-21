package anyllm_test

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"

	anyllm "github.com/mozilla-ai/any-llm/go"
)

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
