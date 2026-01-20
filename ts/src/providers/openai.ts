import OpenAI from "openai";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions";
import { AnyLLM } from "../any_llm.js";
import {
  AuthenticationError,
  ContextLengthExceededError,
  InvalidRequestError,
  MissingApiKeyError,
  ModelNotFoundError,
  ProviderError,
  RateLimitError,
} from "../errors.js";
import type { CompletionOptions, CompletionParams } from "../types.js";

const OPENAI_API_KEY_ENV = "OPENAI_API_KEY";
const PROVIDER_NAME = "openai";

/**
 * OpenAI provider implementation for AnyLLM.
 */
export class OpenAIProvider extends AnyLLM {
  readonly providerName = PROVIDER_NAME;

  private client!: OpenAI;

  /**
   * Initialize the OpenAI client.
   * Falls back to OPENAI_API_KEY environment variable if apiKey is not provided.
   *
   * @param apiKey - Optional API key
   * @param apiBase - Optional API base URL
   * @param options - Additional options (e.g., defaultHeaders)
   */
  initClient(
    apiKey?: string,
    apiBase?: string,
    options?: Record<string, unknown>,
  ): void {
    const resolvedApiKey = apiKey ?? process.env[OPENAI_API_KEY_ENV];

    if (!resolvedApiKey) {
      throw new MissingApiKeyError(PROVIDER_NAME, OPENAI_API_KEY_ENV);
    }

    this.client = new OpenAI({
      apiKey: resolvedApiKey,
      baseURL: apiBase,
      defaultHeaders: options?.defaultHeaders as
        | Record<string, string>
        | undefined,
    });
  }

  /**
   * Perform a chat completion request.
   */
  async completion(
    model: string,
    messages: ChatCompletionMessageParam[],
    options?: CompletionOptions,
  ): Promise<ChatCompletion> {
    const params = this.buildCompletionParams(model, messages, options, false);
    const requestParams = OpenAIProvider.toOpenAIParams(params);

    try {
      const response = await this.client.chat.completions.create({
        ...requestParams,
        stream: false,
      });
      return response;
    } catch (error) {
      throw this.mapError(error);
    }
  }

  /**
   * Perform a streaming chat completion request.
   */
  async *completionStream(
    model: string,
    messages: ChatCompletionMessageParam[],
    options?: CompletionOptions,
  ): AsyncIterable<ChatCompletionChunk> {
    const params = this.buildCompletionParams(model, messages, options, true);
    const requestParams = OpenAIProvider.toOpenAIParams(params);

    try {
      const stream = await this.client.chat.completions.create({
        ...requestParams,
        stream: true,
      });

      for await (const chunk of stream) {
        yield chunk;
      }
    } catch (error) {
      throw this.mapError(error);
    }
  }

  /**
   * Convert CompletionParams to OpenAI-specific request parameters.
   * For OpenAI, the params are already in the correct format.
   */
  private static toOpenAIParams(
    params: CompletionParams,
  ): OpenAI.Chat.ChatCompletionCreateParams {
    // OpenAI uses the standard format, so we can pass through most params
    // Build the params object with only defined values
    const result: OpenAI.Chat.ChatCompletionCreateParams = {
      model: params.model,
      messages: params.messages,
    };

    if (params.temperature !== undefined)
      result.temperature = params.temperature;
    if (params.top_p !== undefined) result.top_p = params.top_p;
    if (params.max_tokens !== undefined) result.max_tokens = params.max_tokens;
    if (params.max_completion_tokens !== undefined)
      result.max_completion_tokens = params.max_completion_tokens;
    if (params.n !== undefined) result.n = params.n;
    if (params.stop !== undefined) result.stop = params.stop;
    if (params.presence_penalty !== undefined)
      result.presence_penalty = params.presence_penalty;
    if (params.frequency_penalty !== undefined)
      result.frequency_penalty = params.frequency_penalty;
    if (params.logit_bias !== undefined) result.logit_bias = params.logit_bias;
    if (params.user !== undefined) result.user = params.user;
    if (params.tools !== undefined) result.tools = params.tools;
    if (params.tool_choice !== undefined)
      result.tool_choice = params.tool_choice;
    if (params.parallel_tool_calls !== undefined)
      result.parallel_tool_calls = params.parallel_tool_calls;
    if (params.response_format !== undefined) {
      result.response_format =
        params.response_format as OpenAI.Chat.ChatCompletionCreateParams["response_format"];
    }
    if (params.logprobs !== undefined) result.logprobs = params.logprobs;
    if (params.top_logprobs !== undefined)
      result.top_logprobs = params.top_logprobs;
    if (params.stream_options !== undefined)
      result.stream_options = params.stream_options;

    return result;
  }

  /**
   * Map OpenAI errors to standardized AnyLLM errors.
   */
  private mapError(error: unknown): Error {
    if (!(error instanceof Error)) {
      return new ProviderError(String(error), undefined, PROVIDER_NAME);
    }

    // Handle OpenAI SDK errors
    if (error instanceof OpenAI.APIError) {
      const status = error.status;
      const message = error.message;

      // Rate limit errors (429)
      if (status === 429) {
        return new RateLimitError(message, error, PROVIDER_NAME);
      }

      // Authentication errors (401)
      if (status === 401) {
        return new AuthenticationError(message, error, PROVIDER_NAME);
      }

      // Invalid request errors (400)
      if (status === 400) {
        // Check for context length errors
        if (
          message.toLowerCase().includes("context length") ||
          message.toLowerCase().includes("maximum context") ||
          message.toLowerCase().includes("token limit")
        ) {
          return new ContextLengthExceededError(message, error, PROVIDER_NAME);
        }
        return new InvalidRequestError(message, error, PROVIDER_NAME);
      }

      // Not found errors (404)
      if (status === 404) {
        // Check for model not found
        if (message.toLowerCase().includes("model")) {
          return new ModelNotFoundError(message, error, PROVIDER_NAME);
        }
        return new InvalidRequestError(message, error, PROVIDER_NAME);
      }

      // Server errors (5xx)
      if (status && status >= 500) {
        return new ProviderError(message, error, PROVIDER_NAME);
      }

      // Other API errors
      return new ProviderError(message, error, PROVIDER_NAME);
    }

    // Generic error handling
    return new ProviderError(error.message, error, PROVIDER_NAME);
  }
}
