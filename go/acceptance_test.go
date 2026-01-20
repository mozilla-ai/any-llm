package anyllm_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	anyllm "github.com/mozilla-ai/any-llm/go"
)

var (
	baseURL     = getEnv("TEST_SERVER_URL", "http://localhost:8080/v1")
	dummyAPIKey = "test-key"
)

func getEnv(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

// TestScenario represents a test scenario from the acceptance test server.
type TestScenario struct {
	Model    string           `json:"model"`
	Messages []anyllm.Message `json:"messages"`
	Stream   bool             `json:"stream,omitempty"`
	Options  map[string]any   `json:"options,omitempty"`
}

// TestRunSummary represents the summary of a test run.
type TestRunSummary struct {
	Total      int            `json:"total"`
	Passed     int            `json:"passed"`
	Failed     int            `json:"failed"`
	ByScenario map[string]any `json:"by_scenario"`
}

// AcceptanceSuite is the test suite for acceptance tests, similar to pytest fixtures.
type AcceptanceSuite struct {
	suite.Suite
	testRunID  string
	serverBase string
	client     anyllm.LLMClient
	scenarios  map[string]TestScenario
}

// SetupSuite runs once before all tests in the suite (like pytest's session-scoped fixtures).
func (s *AcceptanceSuite) SetupSuite() {
	s.testRunID = fmt.Sprintf("go-%d", time.Now().UnixMilli())
	s.serverBase = baseURL[:len(baseURL)-3] // Remove /v1

	// Create test run
	err := s.createTestRun()
	if err != nil {
		s.T().Skipf("Could not create test run (is the acceptance test server running?): %v", err)
	}

	// Load scenarios
	s.scenarios, err = s.loadScenarios()
	require.NoError(s.T(), err, "Failed to load test scenarios")

	// Create client
	s.client, err = anyllm.Create(anyllm.ProviderOpenAI, &anyllm.ClientConfig{
		APIKey:  dummyAPIKey,
		APIBase: baseURL,
		DefaultHeaders: map[string]string{
			"X-Test-Run-Id": s.testRunID,
		},
	})
	require.NoError(s.T(), err, "Failed to create client")
}

// TearDownSuite runs once after all tests complete.
func (s *AcceptanceSuite) TearDownSuite() {
	summary, err := s.getTestRunSummary()
	if err != nil {
		s.T().Logf("Warning: Failed to get test run summary: %v", err)
		return
	}

	assert.Zero(s.T(), summary.Failed, "Test run had %d failures out of %d tests", summary.Failed, summary.Total)
}

// TestScenarios runs all scenario tests dynamically.
func (s *AcceptanceSuite) TestScenarios() {
	for name, scenario := range s.scenarios {
		s.Run(name, func() {
			s.runScenario(scenario)
		})
	}
}

func (s *AcceptanceSuite) runScenario(scenario TestScenario) {
	ctx := context.Background()
	opts := convertOptions(scenario.Options)

	if scenario.Stream {
		s.runStreamingTest(ctx, scenario, opts)
	} else {
		s.runCompletionTest(ctx, scenario, opts)
	}
}

func (s *AcceptanceSuite) runCompletionTest(ctx context.Context, scenario TestScenario, opts *anyllm.CompletionOptions) {
	resp, err := s.client.Completion(ctx, scenario.Model, scenario.Messages, opts)
	require.NoError(s.T(), err, "Completion failed")
	require.NotEmpty(s.T(), resp.Choices, "Response has no choices")

	// Check validation result if present
	if resp.Validation != nil {
		assert.True(s.T(), resp.Validation.Passed,
			"Validation failed for scenario %s: %v", resp.Validation.Scenario, resp.Validation.Errors)
	}
}

func (s *AcceptanceSuite) runStreamingTest(ctx context.Context, scenario TestScenario, opts *anyllm.CompletionOptions) {
	// Ensure stream options are set
	if opts == nil {
		opts = &anyllm.CompletionOptions{}
	}
	includeUsage := true
	opts.StreamOptions = &anyllm.StreamOptions{
		IncludeUsage: &includeUsage,
	}

	chunkChan, errChan := s.client.CompletionStream(ctx, scenario.Model, scenario.Messages, opts)

	var chunks []anyllm.ChatCompletionChunk
	for {
		select {
		case chunk, ok := <-chunkChan:
			if !ok {
				// Channel closed, check for errors
				select {
				case err := <-errChan:
					require.NoError(s.T(), err, "Stream error")
				default:
				}
				goto done
			}
			chunks = append(chunks, chunk)
		case err := <-errChan:
			require.NoError(s.T(), err, "Stream error")
			return
		}
	}

done:
	assert.NotEmpty(s.T(), chunks, "No chunks received in streaming response")
}

func (s *AcceptanceSuite) createTestRun() error {
	url := fmt.Sprintf("%s/v1/test-runs?test_run_id=%s&description=Go%%20acceptance%%20tests", s.serverBase, s.testRunID)
	resp, err := http.Post(url, "application/json", nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusConflict {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}
	return nil
}

func (s *AcceptanceSuite) loadScenarios() (map[string]TestScenario, error) {
	url := fmt.Sprintf("%s/v1/test-data", s.serverBase)
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var data struct {
		Scenarios map[string]TestScenario `json:"scenarios"`
	}
	if err := json.Unmarshal(body, &data); err != nil {
		return nil, err
	}

	return data.Scenarios, nil
}

func (s *AcceptanceSuite) getTestRunSummary() (*TestRunSummary, error) {
	url := fmt.Sprintf("%s/v1/test-runs/%s/summary", s.serverBase, s.testRunID)
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var summary TestRunSummary
	if err := json.Unmarshal(body, &summary); err != nil {
		return nil, err
	}

	return &summary, nil
}

// TestAcceptance is the entry point for running the acceptance test suite.
func TestAcceptance(t *testing.T) {
	suite.Run(t, new(AcceptanceSuite))
}

func convertOptions(opts map[string]any) *anyllm.CompletionOptions {
	if opts == nil || len(opts) == 0 {
		return nil
	}

	result := &anyllm.CompletionOptions{}

	if v, ok := opts["temperature"].(float64); ok {
		result.Temperature = &v
	}
	if v, ok := opts["top_p"].(float64); ok {
		result.TopP = &v
	}
	if v, ok := opts["max_tokens"].(float64); ok {
		i := int(v)
		result.MaxTokens = &i
	}
	if v, ok := opts["presence_penalty"].(float64); ok {
		result.PresencePenalty = &v
	}
	if v, ok := opts["frequency_penalty"].(float64); ok {
		result.FrequencyPenalty = &v
	}
	if v, ok := opts["tools"].([]any); ok && len(v) > 0 {
		result.Tools = convertTools(v)
	}
	if v, ok := opts["response_format"].(map[string]any); ok {
		result.ResponseFormat = convertResponseFormat(v)
	}
	if v, ok := opts["stream_options"].(map[string]any); ok {
		result.StreamOptions = convertStreamOptions(v)
	}

	return result
}

func convertTools(tools []any) []anyllm.Tool {
	result := make([]anyllm.Tool, 0, len(tools))
	for _, t := range tools {
		if toolMap, ok := t.(map[string]any); ok {
			tool := anyllm.Tool{
				Type: "function",
			}
			if fn, ok := toolMap["function"].(map[string]any); ok {
				tool.Function = anyllm.FunctionDefinition{
					Name:        getStringField(fn, "name"),
					Description: getStringField(fn, "description"),
				}
				if params, ok := fn["parameters"].(map[string]any); ok {
					tool.Function.Parameters = params
				}
			}
			result = append(result, tool)
		}
	}
	return result
}

func convertResponseFormat(rf map[string]any) *anyllm.ResponseFormat {
	result := &anyllm.ResponseFormat{
		Type: getStringField(rf, "type"),
	}
	if js, ok := rf["json_schema"].(map[string]any); ok {
		result.JSONSchema = &anyllm.JSONSchemaFormat{
			Name: getStringField(js, "name"),
		}
		if schema, ok := js["schema"].(map[string]any); ok {
			result.JSONSchema.Schema = schema
		}
	}
	return result
}

func convertStreamOptions(so map[string]any) *anyllm.StreamOptions {
	result := &anyllm.StreamOptions{}
	if v, ok := so["include_usage"].(bool); ok {
		result.IncludeUsage = &v
	}
	return result
}

func getStringField(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}
