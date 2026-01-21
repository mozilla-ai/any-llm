import { describe, it, expect } from "vitest";
import { functionToTool } from "../src/tools.js";

describe("functionToTool", () => {
  it("should convert a function to OpenAI tool format", () => {
    function getWeather(location: string): string {
      return `Weather in ${location}`;
    }

    const tool = functionToTool(getWeather, {
      description: "Get weather information for a location.",
      parameters: {
        location: {
          type: { type: "string" },
          description: "The city name to get weather for.",
          required: true,
        },
      },
    });

    expect(tool).toEqual({
      type: "function",
      function: {
        name: "getWeather",
        description: "Get weather information for a location.",
        parameters: {
          type: "object",
          properties: {
            location: {
              type: "string",
              description: "The city name to get weather for.",
            },
          },
          required: ["location"],
        },
      },
    });
  });

  it("should handle optional parameters", () => {
    function getWeather(location: string, unit?: string): string {
      return `Weather in ${location} (${unit ?? "celsius"})`;
    }

    const tool = functionToTool(getWeather, {
      description: "Get weather information for a location.",
      parameters: {
        location: {
          type: { type: "string" },
          description: "The city name to get weather for.",
          required: true,
        },
        unit: {
          type: { type: "string", enum: ["celsius", "fahrenheit"] },
          description: "Temperature unit.",
          required: false,
        },
      },
    });

    expect(tool.function.parameters).toEqual({
      type: "object",
      properties: {
        location: {
          type: "string",
          description: "The city name to get weather for.",
        },
        unit: {
          type: "string",
          enum: ["celsius", "fahrenheit"],
          description: "Temperature unit.",
        },
      },
      required: ["location"],
    });
  });

  it("should use function name from options if provided", () => {
    const func = function () {
      return "result";
    };

    const tool = functionToTool(func, {
      name: "customName",
      description: "A custom function.",
    });

    expect(tool.function.name).toBe("customName");
  });

  it("should generate default description for parameters without one", () => {
    function myFunc(param: string): void {
      console.log(param);
    }

    const tool = functionToTool(myFunc, {
      description: "A function.",
      parameters: {
        param: {
          type: { type: "string" },
        },
      },
    });

    expect(tool.function.parameters?.properties?.param).toEqual({
      type: "string",
      description: "Parameter param of type string",
    });
  });

  it("should throw error if function has no name and options.name is not provided", () => {
    const func = (() => "result") as (...args: unknown[]) => unknown;
    // Arrow functions can have empty name property
    Object.defineProperty(func, "name", { value: "" });

    expect(() =>
      functionToTool(func, {
        description: "A function.",
      }),
    ).toThrow("Function must have a name or options.name must be provided");
  });

  it("should throw error if description is not provided", () => {
    function myFunc(): void {}

    expect(() =>
      functionToTool(myFunc, {
        description: "",
      }),
    ).toThrow("Function description is required");
  });

  it("should handle function with no parameters", () => {
    function getCurrentTime(): string {
      return new Date().toISOString();
    }

    const tool = functionToTool(getCurrentTime, {
      description: "Get the current date and time.",
    });

    expect(tool).toEqual({
      type: "function",
      function: {
        name: "getCurrentTime",
        description: "Get the current date and time.",
        parameters: {
          type: "object",
          properties: {},
          required: [],
        },
      },
    });
  });

  it("should handle complex nested types", () => {
    function searchProducts(
      query: string,
      filters: { category: string; minPrice: number },
    ): string {
      return JSON.stringify({ query, filters });
    }

    const tool = functionToTool(searchProducts, {
      description: "Search for products.",
      parameters: {
        query: {
          type: { type: "string" },
          description: "Search query.",
          required: true,
        },
        filters: {
          type: {
            type: "object",
            properties: {
              category: { type: "string", description: "Product category." },
              minPrice: { type: "number", description: "Minimum price." },
            },
            required: ["category"],
          },
          description: "Search filters.",
          required: true,
        },
      },
    });

    expect(tool.function.parameters?.properties?.filters).toEqual({
      type: "object",
      properties: {
        category: { type: "string", description: "Product category." },
        minPrice: { type: "number", description: "Minimum price." },
      },
      required: ["category"],
      description: "Search filters.",
    });
  });
});
