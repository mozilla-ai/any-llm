import { describe, it, expect } from "vitest";
import { AnyLLM, functionToTool } from "../src/index.js";
import type { ChatCompletion, ChatCompletionMessage } from "../src/index.js";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;

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
  describe("OpenAI provider", () => {
    it(
      "should handle parallel tool calls agent loop",
      async () => {
        if (!OPENAI_API_KEY) {
          console.log("Skipping test: OPENAI_API_KEY not set");
          return;
        }

        const llm = await AnyLLM.create("openai", OPENAI_API_KEY);

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
      },
      60000,
    );
  });

  describe("Anthropic provider", () => {
    it(
      "should handle parallel tool calls agent loop",
      async () => {
        if (!ANTHROPIC_API_KEY) {
          console.log("Skipping test: ANTHROPIC_API_KEY not set");
          return;
        }

        const llm = await AnyLLM.create("anthropic", ANTHROPIC_API_KEY);

        const messages: ChatCompletionMessageParam[] = [
          {
            role: "user",
            content:
              "Get the weather for both Paris and London using the getWeather tool. Call the tool twice, once for each city.",
          },
        ];

        // First completion to get tool calls
        const result: ChatCompletion = await llm.completion(
          "claude-haiku-4-5-20251001",
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
          "claude-haiku-4-5-20251001",
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
      },
      60000,
    );
  });
});
