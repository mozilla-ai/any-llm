/**
 * Acceptance tests for any-llm TypeScript implementation.
 * Validates the client against the acceptance test server.
 *
 * Run with: npx tsx tests/acceptance.test.ts
 *
 * Prerequisites:
 * - Acceptance test server running at http://localhost:8080
 * - Start with: uv run acceptance-tests-server/server.py
 */

import { AnyLLM } from "../src/index.js";
import type { ChatCompletion, ChatCompletionChunk } from "../src/index.js";

const BASE_URL = process.env.TEST_SERVER_URL ?? "http://localhost:8080/v1";
const DUMMY_API_KEY = "test-key";

interface ValidationResult {
  passed: boolean;
  errors: Array<{ field: string; message: string }>;
  scenario: string;
}

interface TestRunSummary {
  test_run_id: string;
  total: number;
  passed: number;
  failed: number;
  by_scenario: Record<string, { passed: number; failed: number }>;
}

type CompletionWithValidation = ChatCompletion & {
  _validation?: ValidationResult;
};

const scenarios = {
  basic_completion: {
    model: "test-basic",
    messages: [{ role: "user" as const, content: "Hello" }],
  },
  tool_calls: {
    model: "test-tools",
    messages: [{ role: "user" as const, content: "What's the weather?" }],
    options: {
      tools: [
        {
          type: "function" as const,
          function: {
            name: "get_weather",
            description: "Get weather for a location",
            parameters: {
              type: "object",
              properties: {
                location: { type: "string" },
              },
              required: ["location"],
            },
          },
        },
      ],
    },
  },
  tool_response: {
    model: "test-tool-response",
    messages: [
      { role: "user" as const, content: "What's the weather in Paris?" },
      {
        role: "assistant" as const,
        content: null,
        tool_calls: [
          {
            id: "call_123",
            type: "function" as const,
            function: {
              name: "get_weather",
              arguments: '{"location": "Paris"}',
            },
          },
        ],
      },
      {
        role: "tool" as const,
        content: '{"temperature": "15C"}',
        tool_call_id: "call_123",
      },
    ],
  },
  streaming: {
    model: "test-stream",
    messages: [{ role: "user" as const, content: "Hello" }],
    stream: true,
    options: {
      streamOptions: { include_usage: true },
    },
  },
  structured_output: {
    model: "test-structured",
    messages: [{ role: "user" as const, content: "Give me a person" }],
    options: {
      responseFormat: {
        type: "json_schema" as const,
        json_schema: {
          name: "person",
          schema: {
            type: "object",
            properties: {
              name: { type: "string" },
              age: { type: "integer" },
            },
          },
        },
      },
    },
  },
  multi_turn: {
    model: "test-multi-turn",
    messages: [
      { role: "user" as const, content: "Hello!" },
      { role: "assistant" as const, content: "Hi there!" },
      { role: "user" as const, content: "How are you?" },
    ],
  },
  system_message: {
    model: "test-system",
    messages: [
      { role: "system" as const, content: "You are a helpful assistant." },
      { role: "user" as const, content: "Hello!" },
    ],
  },
  image_content: {
    model: "test-image",
    messages: [
      {
        role: "user" as const,
        content: [
          { type: "text" as const, text: "What's in this image?" },
          {
            type: "image_url" as const,
            image_url: { url: "https://example.com/image.jpg" },
          },
        ],
      },
    ],
  },
  temperature_params: {
    model: "test-params",
    messages: [{ role: "user" as const, content: "Hello" }],
    options: {
      temperature: 0.7,
      topP: 0.9,
      maxTokens: 100,
      presencePenalty: 0.5,
      frequencyPenalty: 0.5,
    },
  },
};

async function createTestRun(testRunId: string): Promise<void> {
  const serverBase = BASE_URL.replace("/v1", "");
  const url = `${serverBase}/v1/test-runs?test_run_id=${encodeURIComponent(testRunId)}&description=TypeScript%20acceptance%20tests`;
  const response = await fetch(url, { method: "POST" });

  if (!response.ok && response.status !== 409) {
    throw new Error(`Failed to create test run: ${response.status}`);
  }
}

async function getTestRunSummary(testRunId: string): Promise<TestRunSummary> {
  const serverBase = BASE_URL.replace("/v1", "");
  const response = await fetch(
    `${serverBase}/v1/test-runs/${encodeURIComponent(testRunId)}/summary`
  );

  if (!response.ok) {
    throw new Error(`Failed to get test run summary: ${response.status}`);
  }

  return response.json();
}

async function runScenario(
  llm: AnyLLM,
  scenarioName: string,
  scenario: (typeof scenarios)[keyof typeof scenarios],
  testRunId: string
): Promise<{ passed: boolean; error?: string }> {
  console.log(`  Running scenario: ${scenarioName}...`);

  try {
    if ("stream" in scenario && scenario.stream) {
      const streamScenario = scenario as typeof scenarios.streaming;
      const chunks: ChatCompletionChunk[] = [];

      for await (const chunk of llm.completionStream(
        streamScenario.model,
        streamScenario.messages,
        streamScenario.options
      )) {
        chunks.push(chunk);
      }

      if (chunks.length === 0) {
        return { passed: false, error: "No chunks received in streaming response" };
      }

      console.log(`    ✓ Received ${chunks.length} chunks`);
      return { passed: true };
    } else {
      const response = (await llm.completion(
        scenario.model,
        scenario.messages,
        "options" in scenario ? scenario.options : undefined
      )) as CompletionWithValidation;

      const validation = response._validation;

      if (validation) {
        if (validation.passed) {
          console.log(`    ✓ Validation passed`);
          return { passed: true };
        } else {
          const errorMsgs = validation.errors
            .map((e) => `${e.field}: ${e.message}`)
            .join("; ");
          console.log(`    ✗ Validation failed: ${errorMsgs}`);
          return { passed: false, error: errorMsgs };
        }
      }

      if (response.choices && response.choices.length > 0) {
        console.log(`    ✓ Response received (no validation in response)`);
        return { passed: true };
      }

      return { passed: false, error: "Invalid response structure" };
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.log(`    ✗ Error: ${errorMessage}`);
    return { passed: false, error: errorMessage };
  }
}

async function runProviderTests(
  providerName: string,
  testRunId: string
): Promise<{ passed: number; failed: number; results: Record<string, boolean> }> {
  console.log(`\nTesting provider: ${providerName}`);
  console.log("=".repeat(40));

  let llm: AnyLLM;
  try {
    llm = await AnyLLM.create(providerName, DUMMY_API_KEY, BASE_URL, {
      defaultHeaders: { "X-Test-Run-Id": testRunId },
    });
  } catch (error) {
    console.log(
      `  Failed to initialize provider: ${error instanceof Error ? error.message : error}`
    );
    return {
      passed: 0,
      failed: Object.keys(scenarios).length,
      results: Object.fromEntries(
        Object.keys(scenarios).map((k) => [k, false])
      ),
    };
  }

  let passed = 0;
  let failed = 0;
  const results: Record<string, boolean> = {};

  for (const [scenarioName, scenario] of Object.entries(scenarios)) {
    const result = await runScenario(llm, scenarioName, scenario, testRunId);
    results[scenarioName] = result.passed;
    if (result.passed) {
      passed++;
    } else {
      failed++;
    }
  }

  console.log(`\n  Summary: ${passed} passed, ${failed} failed`);
  return { passed, failed, results };
}

async function main(): Promise<void> {
  const testRunId = `ts-${Date.now()}`;
  console.log("any-llm TypeScript Acceptance Tests");
  console.log("====================================");
  console.log(`Test Run ID: ${testRunId}`);
  console.log(`Server URL: ${BASE_URL}`);

  try {
    await createTestRun(testRunId);
    console.log("Test run created successfully.\n");
  } catch (error) {
    console.error(
      `Warning: Could not create test run: ${error instanceof Error ? error.message : error}`
    );
  }

  const providersToTest = ["openai"];
  const totalResults: Record<
    string,
    { passed: number; failed: number; results: Record<string, boolean> }
  > = {};

  for (const provider of providersToTest) {
    totalResults[provider] = await runProviderTests(provider, testRunId);
  }

  console.log("\n" + "=".repeat(60));
  console.log("FINAL RESULTS");
  console.log("=".repeat(60));

  let totalPassed = 0;
  let totalFailed = 0;

  for (const [provider, result] of Object.entries(totalResults)) {
    console.log(`\n${provider}:`);
    console.log(`  Passed: ${result.passed}`);
    console.log(`  Failed: ${result.failed}`);
    totalPassed += result.passed;
    totalFailed += result.failed;
  }

  console.log(`\nTotal: ${totalPassed} passed, ${totalFailed} failed`);

  try {
    const summary = await getTestRunSummary(testRunId);
    console.log("\nServer-side summary:");
    console.log(`  Total: ${summary.total}`);
    console.log(`  Passed: ${summary.passed}`);
    console.log(`  Failed: ${summary.failed}`);
  } catch {
    console.log("\nNote: Could not fetch server-side summary.");
  }

  if (totalFailed > 0) {
    process.exit(1);
  }
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
