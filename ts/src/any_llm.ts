import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions";
import type {
  CompletionOptions,
  CompletionParams,
  LLMProvider,
} from "./types.js";

/**
 * Abstract base class for all LLM providers.
 * All providers must inherit from this class and implement the abstract methods.
 */
export abstract class AnyLLM {
  /**
   * Name of the provider (e.g., "openai", "anthropic").
   */
  abstract readonly providerName: string;

  /**
   * Create a provider-specific AnyLLM instance.
   * Providers are loaded dynamically with lazy imports.
   *
   * @param provider - The provider name or LLMProvider enum value
   * @param apiKey - Optional API key (falls back to environment variable)
   * @param apiBase - Optional API base URL
   * @param options - Additional provider-specific options
   * @returns A provider-specific AnyLLM instance
   */
  static async create(
    provider: string | LLMProvider,
    apiKey?: string,
    apiBase?: string,
    options?: Record<string, unknown>,
  ): Promise<AnyLLM> {
    const providerName = provider.toLowerCase();

    switch (providerName) {
      case "openai": {
        const { OpenAIProvider } = await import("./providers/openai.js");
        const instance = new OpenAIProvider();
        instance.initClient(apiKey, apiBase, options);
        return instance;
      }
      default:
        throw new Error(`Unknown provider: ${provider}`);
    }
  }

  /**
   * Initialize the provider's HTTP/SDK client.
   * When apiKey is undefined, should check for standard environment variables.
   *
   * @param apiKey - Optional API key
   * @param apiBase - Optional API base URL
   * @param options - Additional provider-specific options
   */
  abstract initClient(
    apiKey?: string,
    apiBase?: string,
    options?: Record<string, unknown>,
  ): void;

  /**
   * Perform a chat completion request.
   *
   * @param model - The model ID to use
   * @param messages - The conversation messages
   * @param options - Optional completion parameters
   * @returns The chat completion response
   */
  abstract completion(
    model: string,
    messages: ChatCompletionMessageParam[],
    options?: CompletionOptions,
  ): Promise<ChatCompletion>;

  /**
   * Perform a streaming chat completion request.
   *
   * @param model - The model ID to use
   * @param messages - The conversation messages
   * @param options - Optional completion parameters
   * @returns An async iterable of chat completion chunks
   */
  abstract completionStream(
    model: string,
    messages: ChatCompletionMessageParam[],
    options?: CompletionOptions,
  ): AsyncIterable<ChatCompletionChunk>;

  /**
   * Convert user-friendly CompletionOptions to provider-specific parameters.
   *
   * @param params - The internal completion parameters
   * @param extraOptions - Additional provider-specific options
   * @returns Provider-specific request parameters
   */
  protected static convertCompletionParams(
    params: CompletionParams,
    _extraOptions?: Record<string, unknown>,
  ): Record<string, unknown> {
    return { ...params };
  }

  /**
   * Convert provider-specific response to standard ChatCompletion.
   *
   * @param response - The provider-specific response
   * @returns Standard ChatCompletion response
   */
  protected static convertCompletionResponse(
    response: unknown,
  ): ChatCompletion {
    return response as ChatCompletion;
  }

  /**
   * Convert provider-specific chunk to standard ChatCompletionChunk.
   *
   * @param chunk - The provider-specific chunk
   * @returns Standard ChatCompletionChunk
   */
  protected static convertCompletionChunkResponse(
    chunk: unknown,
  ): ChatCompletionChunk {
    return chunk as ChatCompletionChunk;
  }

  /**
   * Build internal CompletionParams from user options.
   *
   * @param model - The model ID
   * @param messages - The conversation messages
   * @param options - User-friendly completion options
   * @param stream - Whether to enable streaming
   * @returns Internal completion parameters
   */
  protected buildCompletionParams(
    model: string,
    messages: ChatCompletionMessageParam[],
    options?: CompletionOptions,
    stream?: boolean,
  ): CompletionParams {
    const params: CompletionParams = {
      model,
      messages,
    };

    if (stream !== undefined) {
      params.stream = stream;
    }

    if (options) {
      if (options.temperature !== undefined) {
        params.temperature = options.temperature;
      }
      if (options.topP !== undefined) {
        params.top_p = options.topP;
      }
      if (options.maxTokens !== undefined) {
        params.max_tokens = options.maxTokens;
      }
      if (options.maxCompletionTokens !== undefined) {
        params.max_completion_tokens = options.maxCompletionTokens;
      }
      if (options.n !== undefined) {
        params.n = options.n;
      }
      if (options.stop !== undefined) {
        params.stop = options.stop;
      }
      if (options.presencePenalty !== undefined) {
        params.presence_penalty = options.presencePenalty;
      }
      if (options.frequencyPenalty !== undefined) {
        params.frequency_penalty = options.frequencyPenalty;
      }
      if (options.logitBias !== undefined) {
        params.logit_bias = options.logitBias;
      }
      if (options.user !== undefined) {
        params.user = options.user;
      }
      if (options.tools !== undefined) {
        params.tools = options.tools;
      }
      if (options.toolChoice !== undefined) {
        params.tool_choice = options.toolChoice;
      }
      if (options.parallelToolCalls !== undefined) {
        params.parallel_tool_calls = options.parallelToolCalls;
      }
      if (options.responseFormat !== undefined) {
        params.response_format = options.responseFormat;
      }
      if (options.logprobs !== undefined) {
        params.logprobs = options.logprobs;
      }
      if (options.topLogprobs !== undefined) {
        params.top_logprobs = options.topLogprobs;
      }
      if (options.streamOptions !== undefined) {
        params.stream_options = {
          include_usage: options.streamOptions.includeUsage,
        };
      }
    }

    return params;
  }
}
