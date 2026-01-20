/**
 * Acceptance tests for any-llm TypeScript implementation.
 * Validates the client against the acceptance test server.
 *
 * Run with: npm test
 *
 * Prerequisites:
 * - Acceptance test server running at http://localhost:8080
 * - Start with: uv run acceptance-tests-server/server.py
 */

import { describe, it, expect, beforeAll } from "vitest";
import { AnyLLM } from "../src/index.js";
import type { ChatCompletionChunk } from "../src/index.js";

const BASE_URL = process.env.TEST_SERVER_URL ?? "http://localhost:8080/v1";
const DUMMY_API_KEY = "test-key";
const PROVIDERS_TO_TEST = ["openai"];

// Shared state
let testRunId: string;
let scenarios: Record<string, any>;

// Setup fixtures
beforeAll(async () => {
  testRunId = `ts-${Date.now()}`;

  // Load scenarios from server
  const serverBase = BASE_URL.replace("/v1", "");
  const scenariosResponse = await fetch(`${serverBase}/v1/test-data`);
  if (!scenariosResponse.ok) {
    throw new Error(`Failed to load test scenarios: ${scenariosResponse.status}`);
  }
  const data = await scenariosResponse.json();
  scenarios = data.scenarios;

  // Create test run
  const testRunResponse = await fetch(
    `${serverBase}/v1/test-runs?test_run_id=${encodeURIComponent(
      testRunId
    )}&description=TypeScript%20acceptance%20tests`,
    { method: "POST" }
  );

  if (!testRunResponse.ok && testRunResponse.status !== 409) {
    throw new Error(`Failed to create test run: ${testRunResponse.status}`);
  }
});

// Parametrized tests
describe.each(PROVIDERS_TO_TEST)("Provider: %s", (provider) => {
  describe.each(Object.entries(scenarios ?? {}))(
    "Scenario: %s",
    (scenarioName, scenario) => {
      it("should complete successfully", async () => {
        const llm = await AnyLLM.create(provider, DUMMY_API_KEY, BASE_URL, {
          defaultHeaders: { "X-Test-Run-Id": testRunId },
        });

        if (scenario.stream) {
          const chunks: ChatCompletionChunk[] = [];
          const options = scenario.options ?? {};

          for await (const chunk of llm.completionStream(
            scenario.model,
            scenario.messages,
            options
          )) {
            chunks.push(chunk);
          }

          expect(chunks.length).toBeGreaterThan(0);
        } else {
          const options = scenario.options ?? {};
          const response = await llm.completion(
            scenario.model,
            scenario.messages,
            options
          );

          expect(response.choices).toBeDefined();
          expect(response.choices.length).toBeGreaterThan(0);
        }
      });
    }
  );
});
