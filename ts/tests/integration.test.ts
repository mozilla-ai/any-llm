import { describe, it, expect, beforeAll } from "vitest";
import { AnyLLM, functionToTool } from "../src/index.js";
import type { ChatCompletion, ChatCompletionMessage } from "../src/index.js";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";

const BASE_URL = process.env.TEST_SERVER_URL ?? "http://localhost:8080/v1";
const DUMMY_API_KEY = "test-key";

/**
 * Get the weather for a location.
 */
function getWeather(location: string): string {
  return JSON.stringify({
    location,
    temperature: "15C",
    condition: "sunny",
  });
}

const getWeatherTool = functionToTool(getWeather, {
  description: "Get the weather for a location.",
  parameters: {
    location: {
      type: { type: "string" },
      description: "The city name to get weather for.",
      required: true,
    },
  },
});

describe("Integration tests with tool calls", () => {
  let testRunId: string;

  beforeAll(async () => {
    testRunId = `ts-integration-${Date.now()}`;

    const serverBase = BASE_URL.replace("/v1", "");
    const testRunResponse = await fetch(
      `${serverBase}/v1/test-runs?test_run_id=${encodeURIComponent(
        testRunId,
      )}&description=TypeScript%20integration%20tests`,
      { method: "POST" },
    );

    if (!testRunResponse.ok && testRunResponse.status !== 409) {
      throw new Error(`Failed to create test run: ${testRunResponse.status}`);
    }
  });

  describe("OpenAI provider", () => {
    it("should handle parallel tool calls agent loop", async () => {
      const llm = await AnyLLM.create("openai", DUMMY_API_KEY, BASE_URL, {
        defaultHeaders: { "X-Test-Run-Id": testRunId },
      });

      const messages: ChatCompletionMessageParam[] = [
        {
          role: "user",
          content:
            "Get the weather for both Paris and London using the get_weather tool. Call the tool twice, once for each city.",
        },
      ];

      // First completion to get tool calls
      const result: ChatCompletion = await llm.completion(
        "gpt-5-nano",
        messages,
        {
          tools: [getWeatherTool],
        },
      );

      const toolCalls = result.choices[0].message.tool_calls;
      expect(toolCalls).toBeDefined();

      if (!toolCalls) {
        throw new Error("Expected tool calls to be defined");
      }

      expect(toolCalls.length).toBeGreaterThan(0);

      // Add assistant message with tool calls to conversation
      messages.push(result.choices[0].message as ChatCompletionMessageParam);

      // Process each tool call
      for (const toolCall of toolCalls) {
        expect(toolCall.function.name).toBe("getWeather");

        const args = JSON.parse(toolCall.function.arguments) as {
          location: string;
        };
        const toolResult = getWeather(args.location);

        messages.push({
          role: "tool",
          content: toolResult,
          tool_call_id: toolCall.id,
        });
      }

      // Second completion with tool results
      const secondResult: ChatCompletion = await llm.completion(
        "gpt-5-nano",
        messages,
        {
          tools: [getWeatherTool],
        },
      );

      // Should have content or more tool calls
      const secondMessage: ChatCompletionMessage =
        secondResult.choices[0].message;
      expect(
        secondMessage.content !== null ||
          secondMessage.tool_calls !== undefined,
      ).toBe(true);
    });
  });
});
