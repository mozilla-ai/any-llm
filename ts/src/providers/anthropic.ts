import Anthropic from "@anthropic-ai/sdk";
import type {
  ContentBlockStartEvent,
  Message,
  MessageCreateParamsNonStreaming,
  MessageDeltaEvent,
  MessageParam,
  MessageStartEvent,
  MessageStreamEvent,
  RawContentBlockDeltaEvent,
  TextBlock,
  TextBlockParam,
  Tool,
  ToolChoice,
  ToolResultBlockParam,
  ToolUseBlock,
  ToolUseBlockParam,
} from "@anthropic-ai/sdk/resources/messages";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions";
import { AnyLLM } from "../any_llm.js";
import {
  AuthenticationError,
  ContentFilterError,
  ContextLengthExceededError,
  InvalidRequestError,
  MissingApiKeyError,
  ModelNotFoundError,
  ProviderError,
  RateLimitError,
} from "../errors.js";
import type { CompletionOptions, CompletionParams } from "../types.js";

const ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY";
const PROVIDER_NAME = "anthropic";
const DEFAULT_MAX_TOKENS = 8192;

/**
 * Generate a unique ID using crypto.randomUUID().
 * This is safe to use in Node.js 18+ which is the minimum required version.
 */
function generateUUID(): string {
  return crypto.randomUUID();
}

/**
 * Extended ToolChoice type that includes parallel tool use option
 */
type ExtendedToolChoice = ToolChoice & { disable_parallel_tool_use?: boolean };

interface StreamState {
  messageId: string;
  currentContentBlocks: ContentBlockState[];
  currentBlockIndex: number | null;
  createdAt: number;
}

interface ContentBlockState {
  type: string;
  id?: string;
  name?: string;
  arguments?: string;
  text?: string;
}

/**
 * Anthropic provider implementation for AnyLLM.
 */
export class AnthropicProvider extends AnyLLM {
  readonly providerName = PROVIDER_NAME;

  private client!: Anthropic;

  /**
   * Initialize the Anthropic client.
   * Falls back to ANTHROPIC_API_KEY environment variable if apiKey is not provided.
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
    const resolvedApiKey = apiKey ?? process.env[ANTHROPIC_API_KEY_ENV];

    if (!resolvedApiKey) {
      throw new MissingApiKeyError(PROVIDER_NAME, ANTHROPIC_API_KEY_ENV);
    }

    this.client = new Anthropic({
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
    // Check for unsupported response_format
    if (options?.responseFormat) {
      throw new InvalidRequestError(
        "response_format is not supported by Anthropic. Use tool-based JSON extraction instead.",
        undefined,
        PROVIDER_NAME,
      );
    }

    const params = this.buildCompletionParams(model, messages, options, false);
    const anthropicParams = AnthropicProvider.toAnthropicParams(params);

    try {
      const response = await this.client.messages.create(anthropicParams);
      return AnthropicProvider.toCompletionResponse(response as Message, model);
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
    // Check for unsupported response_format
    if (options?.responseFormat) {
      throw new InvalidRequestError(
        "response_format is not supported by Anthropic. Use tool-based JSON extraction instead.",
        undefined,
        PROVIDER_NAME,
      );
    }

    const params = this.buildCompletionParams(model, messages, options, true);
    const anthropicParams = AnthropicProvider.toAnthropicParams(params);

    try {
      // Remove stream from params since we pass it to the stream method
      const { ...streamParams } = anthropicParams;
      const stream = this.client.messages.stream(streamParams);

      // State tracking for streaming
      const state: StreamState = {
        messageId: "",
        currentContentBlocks: [],
        currentBlockIndex: null,
        createdAt: Math.floor(Date.now() / 1000),
      };

      for await (const event of stream) {
        const chunk = AnthropicProvider.toCompletionChunk(event, model, state);
        if (chunk) {
          // Update state based on event type
          AnthropicProvider.updateStreamState(event, state);
          yield chunk;
        }
      }
    } catch (error) {
      throw this.mapError(error);
    }
  }

  /**
   * Convert CompletionParams to Anthropic-specific request parameters.
   */
  private static toAnthropicParams(
    params: CompletionParams,
  ): MessageCreateParamsNonStreaming {
    // Convert messages (extract system messages, convert tool messages)
    const { systemMessages, anthropicMessages } =
      AnthropicProvider.convertMessages(params.messages);

    const result: MessageCreateParamsNonStreaming = {
      model: params.model,
      messages: anthropicMessages,
      max_tokens:
        params.max_tokens ?? params.max_completion_tokens ?? DEFAULT_MAX_TOKENS,
    };

    // Add system message if any
    if (systemMessages.length > 0) {
      result.system = systemMessages.join("\n");
    }

    // Optional parameters
    if (params.temperature !== undefined) {
      result.temperature = params.temperature;
    }
    if (params.top_p !== undefined) {
      result.top_p = params.top_p;
    }
    if (params.stop !== undefined) {
      result.stop_sequences = Array.isArray(params.stop)
        ? params.stop
        : [params.stop];
    }

    // Convert tools
    if (params.tools && params.tools.length > 0) {
      result.tools = params.tools.map(AnthropicProvider.convertTool);
    }

    // Convert tool_choice
    if (params.tool_choice !== undefined) {
      result.tool_choice = AnthropicProvider.convertToolChoice(
        params.tool_choice,
      );
    }

    // Convert parallel_tool_calls (inverted logic)
    if (params.parallel_tool_calls !== undefined && result.tool_choice) {
      const toolChoice = result.tool_choice as ExtendedToolChoice;
      toolChoice.disable_parallel_tool_use = !params.parallel_tool_calls;
    }

    return result;
  }

  /**
   * Convert OpenAI-style messages to Anthropic format.
   */
  private static convertMessages(messages: ChatCompletionMessageParam[]): {
    systemMessages: string[];
    anthropicMessages: MessageParam[];
  } {
    const systemMessages: string[] = [];
    const anthropicMessages: MessageParam[] = [];
    const pendingToolResults: ToolResultBlockParam[] = [];

    for (const msg of messages) {
      const role = msg.role;
      const content = msg.content;

      // Handle system messages
      if (role === "system") {
        if (typeof content === "string") {
          systemMessages.push(content);
        } else if (Array.isArray(content)) {
          for (const part of content) {
            if (typeof part === "object" && part.type === "text") {
              systemMessages.push(part.text);
            }
          }
        }
        continue;
      }

      // Handle tool messages (convert to user message with tool_result)
      if (role === "tool") {
        const toolMsg = msg as {
          role: "tool";
          content: string | null;
          tool_call_id: string;
        };
        const toolResult: ToolResultBlockParam = {
          type: "tool_result",
          tool_use_id: toolMsg.tool_call_id,
          content:
            typeof toolMsg.content === "string"
              ? toolMsg.content
              : JSON.stringify(toolMsg.content),
        };
        pendingToolResults.push(toolResult);
        continue;
      }

      // Flush pending tool results before adding non-tool message
      if (pendingToolResults.length > 0) {
        anthropicMessages.push({
          role: "user",
          content: [...pendingToolResults],
        });
        pendingToolResults.length = 0;
      }

      // Handle assistant messages
      if (role === "assistant") {
        const assistantMsg = AnthropicProvider.convertAssistantMessage(
          msg as ChatCompletionMessageParam & { role: "assistant" },
        );
        anthropicMessages.push(assistantMsg);
        continue;
      }

      // Handle user messages
      if (role === "user") {
        const userMsg = AnthropicProvider.convertUserMessage(
          msg as ChatCompletionMessageParam & { role: "user" },
        );
        anthropicMessages.push(userMsg);
        continue;
      }
    }

    // Flush any remaining tool results
    if (pendingToolResults.length > 0) {
      anthropicMessages.push({
        role: "user",
        content: [...pendingToolResults],
      });
    }

    return { systemMessages, anthropicMessages };
  }

  /**
   * Convert an OpenAI assistant message to Anthropic format.
   */
  private static convertAssistantMessage(
    msg: ChatCompletionMessageParam & { role: "assistant" },
  ): MessageParam {
    const contentBlocks: (TextBlockParam | ToolUseBlockParam)[] = [];

    // Handle text content
    const textContent = msg.content;
    if (textContent) {
      if (typeof textContent === "string") {
        contentBlocks.push({ type: "text", text: textContent });
      } else if (Array.isArray(textContent)) {
        for (const part of textContent) {
          if (typeof part === "object" && part.type === "text") {
            contentBlocks.push({ type: "text", text: part.text });
          }
        }
      }
    }

    // Handle tool calls (convert to tool_use blocks)
    const toolCalls = (
      msg as unknown as { tool_calls?: Array<Record<string, unknown>> }
    ).tool_calls;
    if (toolCalls) {
      for (const tc of toolCalls) {
        const func = tc.function as
          | { name: string; arguments: string }
          | undefined;
        if (!func) continue;

        let args: Record<string, unknown>;
        try {
          args =
            typeof func.arguments === "string"
              ? (JSON.parse(func.arguments) as Record<string, unknown>)
              : (func.arguments as Record<string, unknown>);
        } catch {
          args = {};
        }

        contentBlocks.push({
          type: "tool_use",
          id: (tc.id as string) ?? generateUUID(),
          name: func.name,
          input: args,
        });
      }
    }

    return {
      role: "assistant",
      content:
        contentBlocks.length > 0 ? contentBlocks : [{ type: "text", text: "" }],
    };
  }

  /**
   * Convert an OpenAI user message to Anthropic format.
   */
  private static convertUserMessage(
    msg: ChatCompletionMessageParam & { role: "user" },
  ): MessageParam {
    const content = msg.content;

    if (typeof content === "string") {
      return { role: "user", content };
    }

    // Handle multimodal content
    const contentBlocks: MessageParam["content"] = [];
    if (Array.isArray(content)) {
      for (const part of content) {
        if (typeof part === "object") {
          if (part.type === "text") {
            (contentBlocks as TextBlockParam[]).push({
              type: "text",
              text: part.text,
            });
          } else if (part.type === "image_url") {
            const imageUrl = part.image_url;
            const url =
              typeof imageUrl === "object"
                ? imageUrl.url
                : (imageUrl as string);
            (contentBlocks as unknown[]).push(
              AnthropicProvider.convertImageUrl(url),
            );
          }
        }
      }
    }

    return { role: "user", content: contentBlocks };
  }

  /**
   * Convert an image URL to Anthropic format.
   */
  private static convertImageUrl(url: string): {
    type: "image";
    source:
      | { type: "base64"; media_type: string; data: string }
      | { type: "url"; url: string };
  } {
    // Handle base64 data URLs
    if (url.startsWith("data:")) {
      try {
        const [prefix, data] = url.split(",", 2);
        const mediaType = prefix.split(":")[1].split(";")[0];
        return {
          type: "image",
          source: {
            type: "base64",
            media_type: mediaType,
            data: data,
          },
        };
      } catch {
        // Fall through to URL handling
      }
    }

    // Regular URL
    return {
      type: "image",
      source: {
        type: "url",
        url: url,
      },
    };
  }

  /**
   * Convert OpenAI tool spec to Anthropic format.
   */
  private static convertTool(tool: {
    type: string;
    function: {
      name: string;
      description?: string;
      parameters?: Record<string, unknown>;
    };
  }): Tool {
    const func = tool.function;
    return {
      name: func.name,
      description: func.description ?? "",
      input_schema: (func.parameters ?? {
        type: "object",
        properties: {},
        required: [],
      }) as Tool["input_schema"],
    };
  }

  /**
   * Convert OpenAI tool_choice to Anthropic format.
   */
  private static convertToolChoice(
    toolChoice: string | { type: string; function?: { name: string } },
  ): ToolChoice {
    if (typeof toolChoice === "string") {
      if (toolChoice === "required") {
        return { type: "any" };
      }
      if (toolChoice === "auto") {
        return { type: "auto" };
      }
      if (toolChoice === "none") {
        return { type: "auto" }; // Anthropic doesn't have "none", use auto
      }
      return { type: "auto" };
    }

    if (typeof toolChoice === "object") {
      if (toolChoice.type === "function" && toolChoice.function) {
        return { type: "tool", name: toolChoice.function.name };
      }
    }

    return { type: "auto" };
  }

  /**
   * Convert Anthropic response to ChatCompletion format.
   */
  private static toCompletionResponse(
    response: Message,
    model: string,
  ): ChatCompletion {
    // Convert stop reason
    const finishReason = AnthropicProvider.convertStopReason(
      response.stop_reason,
    );

    // Process content blocks
    let contentText = "";
    const toolCalls: Array<{
      id: string;
      type: "function";
      function: { name: string; arguments: string };
    }> = [];

    for (const block of response.content) {
      if (block.type === "text") {
        contentText += (block as TextBlock).text;
      } else if (block.type === "tool_use") {
        const toolBlock = block as ToolUseBlock;
        toolCalls.push({
          id: toolBlock.id,
          type: "function",
          function: {
            name: toolBlock.name,
            arguments:
              typeof toolBlock.input === "object"
                ? JSON.stringify(toolBlock.input)
                : String(toolBlock.input),
          },
        });
      }
      // Note: ThinkingBlock content is not included in OpenAI-compatible format
    }

    // Build message
    const message: ChatCompletion["choices"][0]["message"] = {
      role: "assistant",
      content: contentText || null,
      refusal: null,
    };

    if (toolCalls.length > 0) {
      message.tool_calls = toolCalls;
    }

    // Build usage
    const usage = {
      prompt_tokens: response.usage.input_tokens,
      completion_tokens: response.usage.output_tokens,
      total_tokens: response.usage.input_tokens + response.usage.output_tokens,
    };

    return {
      id: response.id,
      choices: [
        {
          index: 0,
          message,
          finish_reason: finishReason as
            | "stop"
            | "length"
            | "tool_calls"
            | "content_filter"
            | "function_call",
          logprobs: null,
        },
      ],
      created: Math.floor(Date.now() / 1000),
      model,
      object: "chat.completion",
      usage,
    };
  }

  /**
   * Convert Anthropic stop reason to OpenAI format.
   */
  private static convertStopReason(stopReason: string | null): string {
    if (stopReason === "end_turn") {
      return "stop";
    }
    if (stopReason === "max_tokens") {
      return "length";
    }
    if (stopReason === "tool_use") {
      return "tool_calls";
    }
    if (stopReason === "stop_sequence") {
      return "stop";
    }
    return "stop";
  }

  /**
   * Update stream state based on event type.
   */
  private static updateStreamState(
    event: MessageStreamEvent,
    state: StreamState,
  ): void {
    if (event.type === "message_start") {
      const msgEvent = event as MessageStartEvent;
      state.messageId = msgEvent.message.id;
      state.createdAt = Math.floor(Date.now() / 1000);
    } else if (event.type === "content_block_start") {
      const blockEvent = event as ContentBlockStartEvent;
      state.currentBlockIndex = blockEvent.index;
      const contentBlock = blockEvent.content_block;

      if (contentBlock.type === "tool_use") {
        const toolBlock = contentBlock as ToolUseBlock;
        state.currentContentBlocks.push({
          type: "tool_use",
          id: toolBlock.id,
          name: toolBlock.name,
          arguments: "",
        });
      } else if (contentBlock.type === "text") {
        state.currentContentBlocks.push({
          type: "text",
          text: "",
        });
      }
    } else if (event.type === "content_block_delta") {
      const deltaEvent = event as RawContentBlockDeltaEvent;
      if (
        state.currentBlockIndex !== null &&
        state.currentBlockIndex < state.currentContentBlocks.length
      ) {
        const block = state.currentContentBlocks[state.currentBlockIndex];
        const delta = deltaEvent.delta as {
          type: string;
          text?: string;
          partial_json?: string;
        };
        if (delta.type === "text_delta" && delta.text) {
          block.text = (block.text ?? "") + delta.text;
        } else if (delta.type === "input_json_delta" && delta.partial_json) {
          block.arguments = (block.arguments ?? "") + delta.partial_json;
        }
      }
    }
  }

  /**
   * Convert Anthropic streaming event to ChatCompletionChunk.
   */
  private static toCompletionChunk(
    event: MessageStreamEvent,
    model: string,
    state: StreamState,
  ): ChatCompletionChunk | null {
    const chunkId =
      state.messageId || `chatcmpl-${generateUUID().slice(0, 8)}`;

    if (event.type === "message_start") {
      const msgEvent = event as MessageStartEvent;
      return {
        id: msgEvent.message.id,
        choices: [
          {
            index: 0,
            delta: { role: "assistant", content: "" },
            finish_reason: null,
            logprobs: null,
          },
        ],
        created: state.createdAt,
        model,
        object: "chat.completion.chunk",
      };
    }

    if (event.type === "content_block_start") {
      const blockEvent = event as ContentBlockStartEvent;
      const block = blockEvent.content_block;

      if (block.type === "text") {
        return {
          id: chunkId,
          choices: [
            {
              index: 0,
              delta: { content: "" },
              finish_reason: null,
              logprobs: null,
            },
          ],
          created: state.createdAt,
          model,
          object: "chat.completion.chunk",
        };
      }

      if (block.type === "tool_use") {
        const toolBlock = block as ToolUseBlock;
        return {
          id: chunkId,
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: blockEvent.index,
                    id: toolBlock.id,
                    type: "function" as const,
                    function: {
                      name: toolBlock.name,
                      arguments: "",
                    },
                  },
                ],
              },
              finish_reason: null,
              logprobs: null,
            },
          ],
          created: state.createdAt,
          model,
          object: "chat.completion.chunk",
        };
      }
    }

    if (event.type === "content_block_delta") {
      const deltaEvent = event as RawContentBlockDeltaEvent;
      const delta = deltaEvent.delta as {
        type: string;
        text?: string;
        partial_json?: string;
      };

      if (delta.type === "text_delta" && delta.text) {
        return {
          id: chunkId,
          choices: [
            {
              index: 0,
              delta: { content: delta.text },
              finish_reason: null,
              logprobs: null,
            },
          ],
          created: state.createdAt,
          model,
          object: "chat.completion.chunk",
        };
      }

      if (delta.type === "input_json_delta" && delta.partial_json) {
        return {
          id: chunkId,
          choices: [
            {
              index: 0,
              delta: {
                tool_calls: [
                  {
                    index: state.currentBlockIndex ?? 0,
                    function: {
                      arguments: delta.partial_json,
                    },
                  },
                ],
              },
              finish_reason: null,
              logprobs: null,
            },
          ],
          created: state.createdAt,
          model,
          object: "chat.completion.chunk",
        };
      }
    }

    if (event.type === "content_block_stop") {
      // Check if this was a tool_use block
      if (
        state.currentBlockIndex !== null &&
        state.currentBlockIndex < state.currentContentBlocks.length
      ) {
        const blockState = state.currentContentBlocks[state.currentBlockIndex];
        if (blockState.type === "tool_use") {
          return {
            id: chunkId,
            choices: [
              {
                index: 0,
                delta: {},
                finish_reason: "tool_calls",
                logprobs: null,
              },
            ],
            created: state.createdAt,
            model,
            object: "chat.completion.chunk",
          };
        }
      }
    }

    if (event.type === "message_stop") {
      return {
        id: chunkId,
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: "stop",
            logprobs: null,
          },
        ],
        created: state.createdAt,
        model,
        object: "chat.completion.chunk",
      };
    }

    if (event.type === "message_delta") {
      const msgDeltaEvent = event as MessageDeltaEvent;
      const finishReason = AnthropicProvider.convertStopReason(
        msgDeltaEvent.delta.stop_reason ?? null,
      );
      const usage = msgDeltaEvent.usage
        ? {
            prompt_tokens: 0, // Not available in delta
            completion_tokens: msgDeltaEvent.usage.output_tokens,
            total_tokens: msgDeltaEvent.usage.output_tokens,
          }
        : undefined;

      return {
        id: chunkId,
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: finishReason as
              | "stop"
              | "length"
              | "tool_calls"
              | "content_filter"
              | "function_call"
              | null,
            logprobs: null,
          },
        ],
        created: state.createdAt,
        model,
        object: "chat.completion.chunk",
        usage,
      };
    }

    return null;
  }

  /**
   * Map Anthropic errors to standardized AnyLLM errors.
   */
  private mapError(error: unknown): Error {
    if (!(error instanceof Error)) {
      return new ProviderError(String(error), undefined, PROVIDER_NAME);
    }

    // Handle Anthropic SDK errors
    if (error instanceof Anthropic.APIError) {
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
        const lowerMessage = message.toLowerCase();
        // Check for context length errors
        if (
          lowerMessage.includes("context length") ||
          lowerMessage.includes("maximum context") ||
          lowerMessage.includes("too long")
        ) {
          return new ContextLengthExceededError(message, error, PROVIDER_NAME);
        }
        // Check for content filter errors
        if (
          lowerMessage.includes("content filter") ||
          lowerMessage.includes("content_filter") ||
          lowerMessage.includes("safety")
        ) {
          return new ContentFilterError(message, error, PROVIDER_NAME);
        }
        return new InvalidRequestError(message, error, PROVIDER_NAME);
      }

      // Not found errors (404)
      if (status === 404) {
        return new ModelNotFoundError(message, error, PROVIDER_NAME);
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
