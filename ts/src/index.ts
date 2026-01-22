// Main class
export { AnyLLM } from "./any_llm.js";

// Tools
export { functionToTool } from "./tools.js";
export type {
  FunctionToToolOptions,
  JSONSchemaType,
  ParameterDefinition,
} from "./tools.js";

// Errors
export {
  AnyLLMError,
  AuthenticationError,
  ContentFilterError,
  ContextLengthExceededError,
  InvalidRequestError,
  MissingApiKeyError,
  ModelNotFoundError,
  ProviderError,
  RateLimitError,
} from "./errors.js";

// Types - re-export from OpenAI SDK and custom types
export type {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionMessage,
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
  CompletionOptions,
  CompletionParams,
  LLMProvider,
  ResponseFormat,
  StreamOptions,
} from "./types.js";

// Provider implementations
export { OpenAIProvider } from "./providers/openai.js";
export { AnthropicProvider } from "./providers/anthropic.js";
