/**
 * Re-export OpenAI types through any-llm for a unified import experience.
 */
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionMessage,
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
} from "openai/resources/chat/completions";

// Re-export OpenAI types
export type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionMessage,
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
};

/**
 * Supported LLM provider names.
 */
export type LLMProvider = "openai" | "anthropic" | "google" | "mistral";

/**
 * Completion options interface with camelCase properties for a more TypeScript-friendly API.
 * Maps to the OpenAI API parameters documented in completions.md.
 */
export interface CompletionOptions {
  /**
   * Sampling temperature between 0 and 2. Higher = more random.
   */
  temperature?: number;

  /**
   * Nucleus sampling threshold. 0.1 = only top 10% probability mass.
   */
  topP?: number;

  /**
   * Upper bound for tokens generated.
   * @deprecated Use maxCompletionTokens instead.
   */
  maxTokens?: number;

  /**
   * Upper bound for tokens generated (including reasoning tokens).
   */
  maxCompletionTokens?: number;

  /**
   * Number of completion choices to generate.
   */
  n?: number;

  /**
   * Up to 4 sequences where generation stops.
   */
  stop?: string | string[];

  /**
   * Number between -2.0 and 2.0. Positive values encourage new topics.
   */
  presencePenalty?: number;

  /**
   * Number between -2.0 and 2.0. Positive values penalize repeated tokens.
   */
  frequencyPenalty?: number;

  /**
   * Map of token IDs to bias values (-100 to 100) to modify token likelihood.
   */
  logitBias?: Record<number, number>;

  /**
   * Stable identifier for the user making the request.
   */
  user?: string;

  /**
   * List of tools the model may call.
   */
  tools?: ChatCompletionTool[];

  /**
   * Controls tool calling behavior.
   */
  toolChoice?: ChatCompletionToolChoiceOption;

  /**
   * Whether to enable parallel function calling during tool use.
   */
  parallelToolCalls?: boolean;

  /**
   * Format specification for structured outputs.
   */
  responseFormat?: ResponseFormat;

  /**
   * Whether to return log probabilities of output tokens.
   */
  logprobs?: boolean;

  /**
   * Number of most likely tokens to return (0-20) at each position.
   */
  topLogprobs?: number;

  /**
   * Options for streaming responses.
   */
  streamOptions?: StreamOptions;
}

/**
 * Response format for structured outputs.
 */
export type ResponseFormat =
  | { type: "text" }
  | { type: "json_object" }
  | {
      type: "json_schema";
      json_schema: {
        name: string;
        schema?: Record<string, unknown>;
        description?: string;
        strict?: boolean;
      };
    };

/**
 * Options for streaming responses.
 */
export interface StreamOptions {
  /**
   * Include usage statistics in stream.
   */
  includeUsage?: boolean;
}

/**
 * Internal completion parameters passed to providers.
 * Uses snake_case to match OpenAI API.
 */
export interface CompletionParams {
  model: string;
  messages: ChatCompletionMessageParam[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  max_completion_tokens?: number;
  n?: number;
  stop?: string | string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  logit_bias?: Record<number, number>;
  user?: string;
  tools?: ChatCompletionTool[];
  tool_choice?: ChatCompletionToolChoiceOption;
  parallel_tool_calls?: boolean;
  response_format?: ResponseFormat;
  logprobs?: boolean;
  top_logprobs?: number;
  stream?: boolean;
  stream_options?: { include_usage?: boolean };
}
