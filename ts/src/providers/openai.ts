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
    const requestParams = OpenAIProvider.convertCompletionParams(params);

    try {
      const response = await this.client.chat.completions.create(
        requestParams as unknown as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming,
      );
      return OpenAIProvider.convertCompletionResponse(response);
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
    const requestParams = OpenAIProvider.convertCompletionParams(params);

    try {
      const stream = await this.client.chat.completions.create({
        ...(requestParams as unknown as OpenAI.Chat.ChatCompletionCreateParamsStreaming),
        stream: true,
      });

      for await (const chunk of stream) {
        yield OpenAIProvider.convertCompletionChunkResponse(chunk);
      }
    } catch (error) {
      throw this.mapError(error);
    }
  }

  /**
   * Convert CompletionParams to OpenAI-specific request parameters.
   * For OpenAI, the params are already in the correct format.
   */
  protected static override convertCompletionParams(
    params: CompletionParams,
  ): Record<string, unknown> {
    // OpenAI uses the standard format, so we can pass through most params
    // Filter out undefined values
    const result: Record<string, unknown> = {};

    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined) {
        result[key] = value;
      }
    }

    return result;
  }

  /**
   * Convert OpenAI response to standard ChatCompletion.
   * OpenAI responses are already in the standard format.
   */
  protected static override convertCompletionResponse(
    response: unknown,
  ): ChatCompletion {
    return response as ChatCompletion;
  }

  /**
   * Convert OpenAI chunk to standard ChatCompletionChunk.
   * OpenAI chunks are already in the standard format.
   */
  protected static override convertCompletionChunkResponse(
    chunk: unknown,
  ): ChatCompletionChunk {
    return chunk as ChatCompletionChunk;
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
