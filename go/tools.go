package anyllm

import (
	"reflect"
	"strings"
	"time"
)

// FunctionSpec defines a function that can be converted to an OpenAI tool.
// Since Go doesn't have runtime introspection of function docstrings or parameter names,
// the user must provide this information explicitly.
type FunctionSpec struct {
	// Name is the function name that will be exposed to the LLM.
	Name string

	// Description is the function description for the LLM.
	Description string

	// Parameters defines the function parameters with their types and descriptions.
	Parameters []ParameterSpec

	// Func is an optional reference to the actual function (used for type inference if Parameters not provided).
	Func interface{}
}

// ParameterSpec defines a function parameter.
type ParameterSpec struct {
	// Name is the parameter name.
	Name string

	// Type is the Go type of the parameter.
	Type reflect.Type

	// Description is the parameter description (optional).
	Description string

	// Required indicates if the parameter is required (has no default).
	Required bool

	// Enum provides valid enum values for the parameter (optional).
	Enum []interface{}
}

// FunctionToTool converts a FunctionSpec to OpenAI tool format.
// This function creates a Tool struct that can be passed to the completion API.
//
// Example:
//
//	spec := FunctionSpec{
//	    Name:        "get_weather",
//	    Description: "Get weather information for a location.",
//	    Parameters: []ParameterSpec{
//	        {Name: "location", Type: reflect.TypeOf(""), Description: "The city name", Required: true},
//	        {Name: "unit", Type: reflect.TypeOf(""), Enum: []interface{}{"celsius", "fahrenheit"}},
//	    },
//	}
//	tool := FunctionToTool(spec)
func FunctionToTool(spec FunctionSpec) Tool {
	properties := make(map[string]interface{})
	required := make([]string, 0)

	for _, param := range spec.Parameters {
		paramSchema := typeToJSONSchema(param.Type)

		// Add description
		if param.Description != "" {
			paramSchema["description"] = param.Description
		} else {
			paramSchema["description"] = "Parameter " + param.Name + " of type " + typeNameString(param.Type)
		}

		// Add enum values if provided
		if len(param.Enum) > 0 {
			paramSchema["enum"] = param.Enum
		}

		properties[param.Name] = paramSchema

		if param.Required {
			required = append(required, param.Name)
		}
	}

	parameters := map[string]interface{}{
		"type":       "object",
		"properties": properties,
		"required":   required,
	}

	return Tool{
		Type: "function",
		Function: FunctionDefinition{
			Name:        spec.Name,
			Description: spec.Description,
			Parameters:  parameters,
		},
	}
}

// typeToJSONSchema converts a Go reflect.Type to JSON Schema format.
func typeToJSONSchema(t reflect.Type) map[string]interface{} {
	if t == nil {
		return map[string]interface{}{"type": "string"}
	}

	// Handle pointer types by unwrapping
	if t.Kind() == reflect.Ptr {
		return typeToJSONSchema(t.Elem())
	}

	// Handle primitive types
	switch t.Kind() {
	case reflect.String:
		return map[string]interface{}{"type": "string"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return map[string]interface{}{"type": "integer"}
	case reflect.Float32, reflect.Float64:
		return map[string]interface{}{"type": "number"}
	case reflect.Bool:
		return map[string]interface{}{"type": "boolean"}
	}

	// Handle special string types
	if t == reflect.TypeOf(time.Time{}) {
		return map[string]interface{}{"type": "string", "format": "date-time"}
	}
	if t == reflect.TypeOf([]byte{}) {
		return map[string]interface{}{"type": "string", "contentEncoding": "base64"}
	}

	// Handle slices/arrays
	if t.Kind() == reflect.Slice || t.Kind() == reflect.Array {
		elemSchema := typeToJSONSchema(t.Elem())
		return map[string]interface{}{
			"type":  "array",
			"items": elemSchema,
		}
	}

	// Handle maps
	if t.Kind() == reflect.Map {
		valueSchema := typeToJSONSchema(t.Elem())
		return map[string]interface{}{
			"type":                 "object",
			"additionalProperties": valueSchema,
		}
	}

	// Handle structs
	if t.Kind() == reflect.Struct {
		return structToJSONSchema(t)
	}

	// Default to string for unknown types
	return map[string]interface{}{"type": "string"}
}

// structToJSONSchema converts a Go struct type to JSON Schema object format.
func structToJSONSchema(t reflect.Type) map[string]interface{} {
	properties := make(map[string]interface{})
	required := make([]string, 0)

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)

		// Skip unexported fields
		if !field.IsExported() {
			continue
		}

		// Get JSON tag name, or use field name
		fieldName := field.Name
		if jsonTag := field.Tag.Get("json"); jsonTag != "" {
			parts := strings.Split(jsonTag, ",")
			if parts[0] != "-" && parts[0] != "" {
				fieldName = parts[0]
			}
			// Check for omitempty to determine if required
			hasOmitempty := len(parts) > 1 && containsString(parts[1:], "omitempty")
			if !hasOmitempty {
				required = append(required, fieldName)
			}
		} else {
			// No json tag, assume required
			required = append(required, fieldName)
		}

		properties[fieldName] = typeToJSONSchema(field.Type)
	}

	schema := map[string]interface{}{
		"type":       "object",
		"properties": properties,
	}
	if len(required) > 0 {
		schema["required"] = required
	}
	return schema
}

// containsString checks if a string slice contains a specific string.
func containsString(slice []string, s string) bool {
	for _, v := range slice {
		if v == s {
			return true
		}
	}
	return false
}

// typeNameString returns a human-readable type name for a Go type.
func typeNameString(t reflect.Type) string {
	if t == nil {
		return "string"
	}
	return t.String()
}
