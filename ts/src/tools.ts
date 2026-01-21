/**
 * Tools utilities for converting TypeScript functions to OpenAI tools format.
 *
 * This module provides the `functionToTool` helper that converts a TypeScript
 * function into the OpenAI tool format for LLM function calling.
 *
 * Since TypeScript doesn't have runtime type information like Python does,
 * this implementation requires explicit parameter definitions to be passed.
 */

import type { ChatCompletionTool } from "openai/resources/chat/completions";

/**
 * JSON Schema type definitions for function parameters.
 */
export type JSONSchemaType =
  | { type: "string"; enum?: string[]; description?: string }
  | { type: "integer"; enum?: number[]; description?: string }
  | { type: "number"; enum?: number[]; description?: string }
  | { type: "boolean"; description?: string }
  | {
      type: "array";
      items: JSONSchemaType;
      uniqueItems?: boolean;
      description?: string;
    }
  | {
      type: "object";
      properties?: Record<string, JSONSchemaType>;
      required?: string[];
      additionalProperties?: JSONSchemaType | boolean;
      description?: string;
    };

/**
 * Parameter definition for a function tool.
 */
export interface ParameterDefinition {
  /**
   * The JSON Schema type of the parameter.
   */
  type: JSONSchemaType;
  /**
   * Description of the parameter.
   */
  description?: string;
  /**
   * Whether the parameter is required (no default value).
   */
  required?: boolean;
}

/**
 * Options for converting a function to a tool.
 */
export interface FunctionToToolOptions {
  /**
   * The name of the function. If not provided, uses func.name.
   */
  name?: string;
  /**
   * Description of what the function does.
   */
  description: string;
  /**
   * Parameter definitions for the function.
   */
  parameters?: Record<string, ParameterDefinition>;
}

/**
 * Convert a TypeScript function to OpenAI tools format.
 *
 * Since TypeScript doesn't have runtime type information like Python,
 * parameter definitions must be explicitly provided.
 *
 * @param func - The function to convert
 * @param options - Tool definition options including description and parameters
 * @returns Tool definition in OpenAI format
 *
 * @example
 * ```typescript
 * function getWeather(location: string, unit: string = "celsius"): string {
 *   return `Weather in ${location} is sunny`;
 * }
 *
 * const tool = functionToTool(getWeather, {
 *   description: "Get weather information for a location.",
 *   parameters: {
 *     location: {
 *       type: { type: "string" },
 *       description: "The city name to get weather for.",
 *       required: true,
 *     },
 *     unit: {
 *       type: { type: "string", enum: ["celsius", "fahrenheit"] },
 *       description: "Temperature unit.",
 *       required: false,
 *     },
 *   },
 * });
 * ```
 */
export function functionToTool(
  func: (...args: unknown[]) => unknown,
  options: FunctionToToolOptions,
): ChatCompletionTool {
  const funcName = options.name ?? func.name;

  if (!funcName) {
    throw new Error(
      "Function must have a name or options.name must be provided",
    );
  }

  if (!options.description) {
    throw new Error("Function description is required");
  }

  const properties: Record<string, Record<string, unknown>> = {};
  const required: string[] = [];

  if (options.parameters) {
    for (const [paramName, paramDef] of Object.entries(options.parameters)) {
      const paramSchema: Record<string, unknown> = { ...paramDef.type };

      if (paramDef.description) {
        paramSchema.description = paramDef.description;
      } else {
        // Generate default description like Python implementation
        paramSchema.description = `Parameter ${paramName} of type ${paramDef.type.type}`;
      }

      properties[paramName] = paramSchema;

      if (paramDef.required !== false) {
        required.push(paramName);
      }
    }
  }

  return {
    type: "function",
    function: {
      name: funcName,
      description: options.description,
      parameters: {
        type: "object",
        properties,
        required,
      },
    },
  };
}
