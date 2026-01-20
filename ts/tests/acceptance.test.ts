import { describe, it, expect, beforeAll, beforeEach } from "vitest";
import { AnyLLM } from "../src/index.js";
import type { ChatCompletionChunk } from "../src/index.js";

const BASE_URL = process.env.TEST_SERVER_URL ?? "http://localhost:8080/v1";
const DUMMY_API_KEY = "test-key";
const PROVIDERS_TO_TEST = ["openai"];

interface Scenario {
  model: string;
  messages: Array<{ role: string; content: string | object[] }>;
  stream: boolean;
  options: Record<string, unknown>;
}

let testRunId: string;
let scenarios: Record<string, Scenario> = {};

beforeAll(async () => {
  testRunId = `ts-${Date.now()}`;

  const serverBase = BASE_URL.replace("/v1", "");
  const scenariosResponse = await fetch(`${serverBase}/v1/test-data`);
  if (!scenariosResponse.ok) {
    throw new Error(
      `Failed to load test scenarios: ${scenariosResponse.status}`,
    );
  }
  const data = await scenariosResponse.json();
  scenarios = data.scenarios;

  const testRunResponse = await fetch(
    `${serverBase}/v1/test-runs?test_run_id=${encodeURIComponent(
      testRunId,
    )}&description=TypeScript%20acceptance%20tests`,
    { method: "POST" },
  );

  if (!testRunResponse.ok && testRunResponse.status !== 409) {
    throw new Error(`Failed to create test run: ${testRunResponse.status}`);
  }
});

describe.each(PROVIDERS_TO_TEST)("Provider: %s", (provider) => {
  let llm: AnyLLM;

  beforeEach(async () => {
    llm = await AnyLLM.create(provider, DUMMY_API_KEY, BASE_URL, {
      defaultHeaders: { "X-Test-Run-Id": testRunId },
    });
  });

  it("should complete basic_completion scenario", async () => {
    const scenario = scenarios["basic_completion"];
    expect(scenario).toBeDefined();

    const response = await llm.completion(
      scenario.model,
      scenario.messages,
      scenario.options,
    );

    expect(response.choices).toBeDefined();
    expect(response.choices.length).toBeGreaterThan(0);
  });

  it("should complete tool_calls scenario", async () => {
    const scenario = scenarios["tool_calls"];
    expect(scenario).toBeDefined();

    const response = await llm.completion(
      scenario.model,
      scenario.messages,
      scenario.options,
    );

    expect(response.choices).toBeDefined();
    expect(response.choices.length).toBeGreaterThan(0);
  });

  it("should complete tool_response scenario", async () => {
    const scenario = scenarios["tool_response"];
    expect(scenario).toBeDefined();

    const response = await llm.completion(
      scenario.model,
      scenario.messages,
      scenario.options,
    );

    expect(response.choices).toBeDefined();
    expect(response.choices.length).toBeGreaterThan(0);
  });

  it("should complete streaming scenario", async () => {
    const scenario = scenarios["streaming"];
    expect(scenario).toBeDefined();

    const chunks: ChatCompletionChunk[] = [];
    for await (const chunk of llm.completionStream(
      scenario.model,
      scenario.messages,
      scenario.options,
    )) {
      chunks.push(chunk);
    }

    expect(chunks.length).toBeGreaterThan(0);
  });

  it("should complete structured_output scenario", async () => {
    const scenario = scenarios["structured_output"];
    expect(scenario).toBeDefined();

    const response = await llm.completion(
      scenario.model,
      scenario.messages,
      scenario.options,
    );

    expect(response.choices).toBeDefined();
    expect(response.choices.length).toBeGreaterThan(0);
  });

  it("should complete multi_turn scenario", async () => {
    const scenario = scenarios["multi_turn"];
    expect(scenario).toBeDefined();

    const response = await llm.completion(
      scenario.model,
      scenario.messages,
      scenario.options,
    );

    expect(response.choices).toBeDefined();
    expect(response.choices.length).toBeGreaterThan(0);
  });

  it("should complete system_message scenario", async () => {
    const scenario = scenarios["system_message"];
    expect(scenario).toBeDefined();

    const response = await llm.completion(
      scenario.model,
      scenario.messages,
      scenario.options,
    );

    expect(response.choices).toBeDefined();
    expect(response.choices.length).toBeGreaterThan(0);
  });

  it("should complete image_content scenario", async () => {
    const scenario = scenarios["image_content"];
    expect(scenario).toBeDefined();

    const response = await llm.completion(
      scenario.model,
      scenario.messages,
      scenario.options,
    );

    expect(response.choices).toBeDefined();
    expect(response.choices.length).toBeGreaterThan(0);
  });

  it("should complete temperature_params scenario", async () => {
    const scenario = scenarios["temperature_params"];
    expect(scenario).toBeDefined();

    const response = await llm.completion(
      scenario.model,
      scenario.messages,
      scenario.options,
    );

    expect(response.choices).toBeDefined();
    expect(response.choices.length).toBeGreaterThan(0);
  });
});
