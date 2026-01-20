/**
 * Base error class for all any-llm errors.
 * Preserves the original exception from the provider SDK.
 */
export class AnyLLMError extends Error {
  static readonly defaultMessage: string = "An error occurred";

  readonly originalException: Error | undefined;
  readonly providerName: string | undefined;

  constructor(
    message?: string,
    originalException?: Error,
    providerName?: string,
  ) {
    super(message ?? AnyLLMError.defaultMessage);
    this.name = "AnyLLMError";
    this.originalException = originalException;
    this.providerName = providerName;
  }

  override toString(): string {
    if (this.providerName) {
      return `[${this.providerName}] ${this.message}`;
    }
    return this.message;
  }
}

/**
 * Raised when the API rate limit is exceeded.
 */
export class RateLimitError extends AnyLLMError {
  static override readonly defaultMessage = "Rate limit exceeded";

  constructor(
    message?: string,
    originalException?: Error,
    providerName?: string,
  ) {
    super(
      message ?? RateLimitError.defaultMessage,
      originalException,
      providerName,
    );
    this.name = "RateLimitError";
  }
}

/**
 * Raised when authentication with the provider fails.
 */
export class AuthenticationError extends AnyLLMError {
  static override readonly defaultMessage = "Authentication failed";

  constructor(
    message?: string,
    originalException?: Error,
    providerName?: string,
  ) {
    super(
      message ?? AuthenticationError.defaultMessage,
      originalException,
      providerName,
    );
    this.name = "AuthenticationError";
  }
}

/**
 * Raised when the request to the provider is invalid.
 */
export class InvalidRequestError extends AnyLLMError {
  static override readonly defaultMessage = "Invalid request";

  constructor(
    message?: string,
    originalException?: Error,
    providerName?: string,
  ) {
    super(
      message ?? InvalidRequestError.defaultMessage,
      originalException,
      providerName,
    );
    this.name = "InvalidRequestError";
  }
}

/**
 * Raised when the provider encounters an internal error.
 */
export class ProviderError extends AnyLLMError {
  static override readonly defaultMessage = "Provider error";

  constructor(
    message?: string,
    originalException?: Error,
    providerName?: string,
  ) {
    super(
      message ?? ProviderError.defaultMessage,
      originalException,
      providerName,
    );
    this.name = "ProviderError";
  }
}

/**
 * Raised when content is blocked by the provider's safety filter.
 */
export class ContentFilterError extends AnyLLMError {
  static override readonly defaultMessage = "Content blocked by safety filter";

  constructor(
    message?: string,
    originalException?: Error,
    providerName?: string,
  ) {
    super(
      message ?? ContentFilterError.defaultMessage,
      originalException,
      providerName,
    );
    this.name = "ContentFilterError";
  }
}

/**
 * Raised when the requested model is not found or not available.
 */
export class ModelNotFoundError extends AnyLLMError {
  static override readonly defaultMessage = "Model not found";

  constructor(
    message?: string,
    originalException?: Error,
    providerName?: string,
  ) {
    super(
      message ?? ModelNotFoundError.defaultMessage,
      originalException,
      providerName,
    );
    this.name = "ModelNotFoundError";
  }
}

/**
 * Raised when the input exceeds the model's maximum context length.
 */
export class ContextLengthExceededError extends AnyLLMError {
  static override readonly defaultMessage = "Context length exceeded";

  constructor(
    message?: string,
    originalException?: Error,
    providerName?: string,
  ) {
    super(
      message ?? ContextLengthExceededError.defaultMessage,
      originalException,
      providerName,
    );
    this.name = "ContextLengthExceededError";
  }
}

/**
 * Raised when a required API key is not provided.
 */
export class MissingApiKeyError extends AnyLLMError {
  readonly envVarName: string;

  constructor(providerName: string, envVarName: string) {
    const message =
      `No ${providerName} API key provided. ` +
      `Please provide it in the config or set the ${envVarName} environment variable.`;
    super(message, undefined, providerName);
    this.name = "MissingApiKeyError";
    this.envVarName = envVarName;
  }
}
