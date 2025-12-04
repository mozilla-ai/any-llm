from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from any_llm import AnyLLM
from any_llm.constants import LLMProvider
from any_llm.types.batch import Batch
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage, CreateEmbeddingResponse
from any_llm.types.model import Model
from any_llm.types.responses import Response, ResponseInputParam, ResponseStreamEvent
from any_llm.utils.decorators import BATCH_API_EXPERIMENTAL_MESSAGE, experimental


def completion(
    model: str,
    messages: list[dict[str, Any] | ChatCompletionMessage],
    *,
    provider: str | LLMProvider | None = None,
    tools: list[dict[str, Any] | Callable[..., Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | type[BaseModel] | None = None,
    stream: bool | None = None,
    n: int | None = None,
    stop: str | list[str] | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    seed: int | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    user: str | None = None,
    parallel_tool_calls: bool | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    logit_bias: dict[str, float] | None = None,
    stream_options: dict[str, Any] | None = None,
    max_completion_tokens: int | None = None,
    reasoning_effort: Literal["minimal", "low", "medium", "high", "auto"] | None = "auto",
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ChatCompletion | Iterator[ChatCompletionChunk]:
    """Create a chat completion.

    Args:
        model: Model identifier. **Recommended**: Use with separate `provider` parameter (e.g., model='gpt-4', provider='openai').
            **Alternative**: Combined format 'provider:model' (e.g., 'openai:gpt-4').
            Legacy format 'provider/model' is also supported but deprecated.
        provider: **Recommended**: Provider name to use for the request (e.g., 'openai', 'mistral').
            When provided, the model parameter should contain only the model name.
        messages: List of messages for the conversation
        tools: List of tools for tool calling. Can be Python callables or OpenAI tool format dicts
        tool_choice: Controls which tools the model can call
        temperature: Controls randomness in the response (0.0 to 2.0)
        top_p: Controls diversity via nucleus sampling (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        response_format: Format specification for the response
        stream: Whether to stream the response
        n: Number of completions to generate
        stop: Stop sequences for generation
        presence_penalty: Penalize new tokens based on presence in text
        frequency_penalty: Penalize new tokens based on frequency in text
        seed: Random seed for reproducible results
        api_key: API key for the provider
        api_base: Base URL for the provider API
        user: Unique identifier for the end user
        parallel_tool_calls: Whether to allow parallel tool calls
        logprobs: Include token-level log probabilities in the response
        top_logprobs: Number of alternatives to return when logprobs are requested
        logit_bias: Bias the likelihood of specified tokens during generation
        stream_options: Additional options controlling streaming behavior
        max_completion_tokens: Maximum number of tokens for the completion
        reasoning_effort: Reasoning effort level for models that support it. "auto" will map to each provider's default.
        client_args: Additional provider-specific arguments that will be passed to the provider's client instantiation.
        **kwargs: Additional provider-specific arguments that will be passed to the provider's API call.

    Returns:
        The completion response from the provider

    """
    if provider is None:
        provider_key, model_id = AnyLLM.split_model_provider(model)
    else:
        provider_key = LLMProvider.from_string(provider)
        model_id = model

    llm = AnyLLM.create(
        provider_key,
        api_key=api_key,
        api_base=api_base,
        **client_args or {},
    )
    return llm.completion(
        model=model_id,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        response_format=response_format,
        stream=stream,
        n=n,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        seed=seed,
        user=user,
        parallel_tool_calls=parallel_tool_calls,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        logit_bias=logit_bias,
        stream_options=stream_options,
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort,
        **kwargs,
    )


async def acompletion(
    model: str,
    messages: list[dict[str, Any] | ChatCompletionMessage],
    *,
    provider: str | LLMProvider | None = None,
    tools: list[dict[str, Any] | Callable[..., Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | type[BaseModel] | None = None,
    stream: bool | None = None,
    n: int | None = None,
    stop: str | list[str] | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    seed: int | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    user: str | None = None,
    parallel_tool_calls: bool | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    logit_bias: dict[str, float] | None = None,
    stream_options: dict[str, Any] | None = None,
    max_completion_tokens: int | None = None,
    reasoning_effort: Literal["minimal", "low", "medium", "high", "auto"] | None = "auto",
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ChatCompletion | AsyncIterator[ChatCompletionChunk]:
    """Create a chat completion asynchronously.

    Args:
        model: Model identifier. **Recommended**: Use with separate `provider` parameter (e.g., model='gpt-4', provider='openai').
            **Alternative**: Combined format 'provider:model' (e.g., 'openai:gpt-4').
            Legacy format 'provider/model' is also supported but deprecated.
        provider: **Recommended**: Provider name to use for the request (e.g., 'openai', 'mistral').
            When provided, the model parameter should contain only the model name.
        messages: List of messages for the conversation
        tools: List of tools for tool calling. Can be Python callables or OpenAI tool format dicts
        tool_choice: Controls which tools the model can call
        temperature: Controls randomness in the response (0.0 to 2.0)
        top_p: Controls diversity via nucleus sampling (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate
        response_format: Format specification for the response
        stream: Whether to stream the response
        n: Number of completions to generate
        stop: Stop sequences for generation
        presence_penalty: Penalize new tokens based on presence in text
        frequency_penalty: Penalize new tokens based on frequency in text
        seed: Random seed for reproducible results
        api_key: API key for the provider
        api_base: Base URL for the provider API
        user: Unique identifier for the end user
        parallel_tool_calls: Whether to allow parallel tool calls
        logprobs: Include token-level log probabilities in the response
        top_logprobs: Number of alternatives to return when logprobs are requested
        logit_bias: Bias the likelihood of specified tokens during generation
        stream_options: Additional options controlling streaming behavior
        max_completion_tokens: Maximum number of tokens for the completion
        reasoning_effort: Reasoning effort level for models that support it. "auto" will map to each provider's default.
        client_args: Additional provider-specific arguments that will be passed to the provider's client instantiation.
        **kwargs: Additional provider-specific arguments that will be passed to the provider's API call.

    Returns:
        The completion response from the provider

    """
    if provider is None:
        provider_key, model_id = AnyLLM.split_model_provider(model)
    else:
        provider_key = LLMProvider.from_string(provider)
        model_id = model

    llm = AnyLLM.create(
        provider_key,
        api_key=api_key,
        api_base=api_base,
        **client_args or {},
    )
    return await llm.acompletion(
        model=model_id,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        response_format=response_format,
        stream=stream,
        n=n,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        seed=seed,
        user=user,
        parallel_tool_calls=parallel_tool_calls,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        logit_bias=logit_bias,
        stream_options=stream_options,
        max_completion_tokens=max_completion_tokens,
        reasoning_effort=reasoning_effort,
        **kwargs,
    )


def responses(
    model: str,
    input_data: str | ResponseInputParam,
    *,
    provider: str | LLMProvider | None = None,
    tools: list[dict[str, Any] | Callable[..., Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    max_tool_calls: int | None = None,
    parallel_tool_calls: int | None = None,
    reasoning: Any | None = None,
    text: Any | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Response | Iterator[ResponseStreamEvent]:
    """Create a response using the OpenAI-style Responses API.

    This follows the OpenAI Responses API shape and returns the aliased
    `any_llm.types.responses.Response` type. If `stream=True`, an iterator of
    `any_llm.types.responses.ResponseStreamEvent` items is returned.

    Args:
        model: Model identifier in format 'provider/model' (e.g., 'openai/gpt-4o'). If provider is provided, we assume that the model does not contain the provider name. Otherwise, we assume that the model contains the provider name, like 'openai/gpt-4o'.
        provider: Provider name to use for the request. If provided, we assume that the model does not contain the provider name. Otherwise, we assume that the model contains the provider name, like 'openai:gpt-4o'.
        input_data: The input payload accepted by provider's Responses API.
            For OpenAI-compatible providers, this is typically a list mixing
            text, images, and tool instructions, or a dict per OpenAI spec.
        tools: Optional tools for tool calling (Python callables or OpenAI tool dicts)
        tool_choice: Controls which tools the model can call
        max_output_tokens: Maximum number of output tokens to generate
        temperature: Controls randomness in the response (0.0 to 2.0)
        top_p: Controls diversity via nucleus sampling (0.0 to 1.0)
        stream: Whether to stream response events
        api_key: API key for the provider
        api_base: Base URL for the provider API
        instructions: A system (or developer) message inserted into the model's context.
        max_tool_calls: The maximum number of total calls to built-in tools that can be processed in a response. This maximum number applies across all built-in tool calls, not per individual tool. Any further attempts to call a tool by the model will be ignored.
        parallel_tool_calls: Whether to allow the model to run tool calls in parallel.
        reasoning: Configuration options for reasoning models.
        text: Configuration options for a text response from the model. Can be plain text or structured JSON data.
        client_args: Additional provider-specific arguments that will be passed to the provider's client instantiation.
        **kwargs: Additional provider-specific arguments that will be passed to the provider's API call.

    Returns:
        Either a `Response` object (non-streaming) or an iterator of
        `ResponseStreamEvent` (streaming).

    Raises:
        NotImplementedError: If the selected provider does not support the Responses API.

    """
    if provider is None:
        provider_key, model_id = AnyLLM.split_model_provider(model)
    else:
        provider_key = LLMProvider.from_string(provider)
        model_id = model

    llm = AnyLLM.create(
        provider_key,
        api_key=api_key,
        api_base=api_base,
        **client_args or {},
    )
    return llm.responses(
        model=model_id,
        input_data=input_data,
        tools=tools,
        tool_choice=tool_choice,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
        instructions=instructions,
        max_tool_calls=max_tool_calls,
        parallel_tool_calls=parallel_tool_calls,
        reasoning=reasoning,
        text=text,
        **kwargs,
    )


async def aresponses(
    model: str,
    input_data: str | ResponseInputParam,
    *,
    provider: str | LLMProvider | None = None,
    tools: list[dict[str, Any] | Callable[..., Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    instructions: str | None = None,
    max_tool_calls: int | None = None,
    parallel_tool_calls: int | None = None,
    reasoning: Any | None = None,
    text: Any | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Response | AsyncIterator[ResponseStreamEvent]:
    """Create a response using the OpenAI-style Responses API.

    This follows the OpenAI Responses API shape and returns the aliased
    `any_llm.types.responses.Response` type. If `stream=True`, an iterator of
    `any_llm.types.responses.ResponseStreamEvent` items is returned.

    Args:
        model: Model identifier in format 'provider/model' (e.g., 'openai/gpt-4o'). If provider is provided, we assume that the model does not contain the provider name. Otherwise, we assume that the model contains the provider name, like 'openai/gpt-4o'.
        provider: Provider name to use for the request. If provided, we assume that the model does not contain the provider name. Otherwise, we assume that the model contains the provider name, like 'openai:gpt-4o'.
        input_data: The input payload accepted by provider's Responses API.
            For OpenAI-compatible providers, this is typically a list mixing
            text, images, and tool instructions, or a dict per OpenAI spec.
        tools: Optional tools for tool calling (Python callables or OpenAI tool dicts)
        tool_choice: Controls which tools the model can call
        max_output_tokens: Maximum number of output tokens to generate
        temperature: Controls randomness in the response (0.0 to 2.0)
        top_p: Controls diversity via nucleus sampling (0.0 to 1.0)
        stream: Whether to stream response events
        api_key: API key for the provider
        api_base: Base URL for the provider API
        instructions: A system (or developer) message inserted into the model's context.
        max_tool_calls: The maximum number of total calls to built-in tools that can be processed in a response. This maximum number applies across all built-in tool calls, not per individual tool. Any further attempts to call a tool by the model will be ignored.
        parallel_tool_calls: Whether to allow the model to run tool calls in parallel.
        reasoning: Configuration options for reasoning models.
        text: Configuration options for a text response from the model. Can be plain text or structured JSON data.
        client_args: Additional provider-specific arguments that will be passed to the provider's client instantiation.
        **kwargs: Additional provider-specific arguments that will be passed to the provider's API call.

    Returns:
        Either a `Response` object (non-streaming) or an iterator of
        `ResponseStreamEvent` (streaming).

    Raises:
        NotImplementedError: If the selected provider does not support the Responses API.

    """
    if provider is None:
        provider_key, model_id = AnyLLM.split_model_provider(model)
    else:
        provider_key = LLMProvider.from_string(provider)
        model_id = model

    llm = AnyLLM.create(
        provider_key,
        api_key=api_key,
        api_base=api_base,
        **client_args or {},
    )
    return await llm.aresponses(
        model=model_id,
        input_data=input_data,
        tools=tools,
        tool_choice=tool_choice,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
        instructions=instructions,
        max_tool_calls=max_tool_calls,
        parallel_tool_calls=parallel_tool_calls,
        reasoning=reasoning,
        text=text,
        **kwargs,
    )


def embedding(
    model: str,
    inputs: str | list[str],
    *,
    provider: str | LLMProvider | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CreateEmbeddingResponse:
    """Create an embedding.

    Args:
        model: Model identifier. **Recommended**: Use with separate `provider` parameter (e.g., model='gpt-4', provider='openai').
            **Alternative**: Combined format 'provider:model' (e.g., 'openai:gpt-4').
            Legacy format 'provider/model' is also supported but deprecated.
        provider: **Recommended**: Provider name to use for the request (e.g., 'openai', 'mistral').
            When provided, the model parameter should contain only the model name.
        inputs: The input text to embed
        api_key: API key for the provider
        api_base: Base URL for the provider API
        client_args: Additional provider-specific arguments that will be passed to the provider's client instantiation.
        **kwargs: Additional provider-specific arguments that will be passed to the provider's API call.

    Returns:
        The embedding of the input text

    """
    if provider is None:
        provider_key, model_name = AnyLLM.split_model_provider(model)
    else:
        provider_key = LLMProvider.from_string(provider)
        model_name = model

    llm = AnyLLM.create(provider_key, api_key=api_key, api_base=api_base, **client_args or {})
    return llm._embedding(model_name, inputs, **kwargs)


async def aembedding(
    model: str,
    inputs: str | list[str],
    *,
    provider: str | LLMProvider | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> CreateEmbeddingResponse:
    """Create an embedding asynchronously.

    Args:
        model: Model identifier in format 'provider/model' (e.g., 'openai/text-embedding-3-small'). If provider is provided, we assume that the model does not contain the provider name. Otherwise, we assume that the model contains the provider name, like 'openai/gpt-4o'.
        provider: Provider name to use for the request. If provided, we assume that the model does not contain the provider name. Otherwise, we assume that the model contains the provider name, like 'openai:gpt-4o'.
        inputs: The input text to embed
        api_key: API key for the provider
        api_base: Base URL for the provider API
        client_args: Additional provider-specific arguments that will be passed to the provider's client instantiation.
        **kwargs: Additional provider-specific arguments that will be passed to the provider's API call.

    Returns:
        The embedding of the input text

    """
    if provider is None:
        provider_key, model_name = AnyLLM.split_model_provider(model)
    else:
        provider_key = LLMProvider.from_string(provider)
        model_name = model

    llm = AnyLLM.create(provider_key, api_key=api_key, api_base=api_base, **client_args or {})
    return await llm._aembedding(model_name, inputs, **kwargs)


def list_models(
    provider: str | LLMProvider,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Sequence[Model]:
    """List available models for a provider."""
    llm = AnyLLM.create(LLMProvider.from_string(provider), api_key=api_key, api_base=api_base, **client_args or {})
    return llm.list_models(**kwargs)


async def alist_models(
    provider: str | LLMProvider,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Sequence[Model]:
    """List available models for a provider asynchronously."""
    llm = AnyLLM.create(LLMProvider.from_string(provider), api_key=api_key, api_base=api_base, **client_args or {})
    return await llm.alist_models(**kwargs)


@experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
def create_batch(
    provider: str | LLMProvider,
    input_file_path: str,
    endpoint: str,
    *,
    completion_window: str = "24h",
    metadata: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Batch:
    """Create a batch job.

    Args:
        provider: Provider name to use for the request (e.g., 'openai', 'mistral')
        input_file_path: Path to a local file containing batch requests in JSONL format.
        endpoint: The endpoint to be used for all requests (e.g., '/v1/chat/completions')
        completion_window: The time frame within which the batch should be processed (default: '24h')
        metadata: Optional custom metadata for the batch
        api_key: API key for the provider
        api_base: Base URL for the provider API
        client_args: Additional provider-specific arguments for client instantiation
        **kwargs: Additional provider-specific arguments

    Returns:
        The created batch object

    """
    llm = AnyLLM.create(LLMProvider.from_string(provider), api_key=api_key, api_base=api_base, **client_args or {})
    return llm.create_batch(
        input_file_path=input_file_path,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata,
        **kwargs,
    )


@experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
async def acreate_batch(
    provider: str | LLMProvider,
    input_file_path: str,
    endpoint: str,
    *,
    completion_window: str = "24h",
    metadata: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Batch:
    """Create a batch job asynchronously.

    Args:
        provider: Provider name to use for the request (e.g., 'openai', 'mistral')
        input_file_path: Path to a local file containing batch requests in JSONL format.
        endpoint: The endpoint to be used for all requests (e.g., '/v1/chat/completions')
        completion_window: The time frame within which the batch should be processed (default: '24h')
        metadata: Optional custom metadata for the batch
        api_key: API key for the provider
        api_base: Base URL for the provider API
        client_args: Additional provider-specific arguments for client instantiation
        **kwargs: Additional provider-specific arguments

    Returns:
        The created batch object

    """
    llm = AnyLLM.create(LLMProvider.from_string(provider), api_key=api_key, api_base=api_base, **client_args or {})
    return await llm.acreate_batch(
        input_file_path=input_file_path,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata,
        **kwargs,
    )


@experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
def retrieve_batch(
    provider: str | LLMProvider,
    batch_id: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Batch:
    """Retrieve a batch job.

    Args:
        provider: Provider name to use for the request (e.g., 'openai', 'mistral')
        batch_id: The ID of the batch to retrieve
        api_key: API key for the provider
        api_base: Base URL for the provider API
        client_args: Additional provider-specific arguments for client instantiation
        **kwargs: Additional provider-specific arguments

    Returns:
        The batch object

    """
    llm = AnyLLM.create(LLMProvider.from_string(provider), api_key=api_key, api_base=api_base, **client_args or {})
    return llm.retrieve_batch(batch_id, **kwargs)


@experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
async def aretrieve_batch(
    provider: str | LLMProvider,
    batch_id: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Batch:
    """Retrieve a batch job asynchronously.

    Args:
        provider: Provider name to use for the request (e.g., 'openai', 'mistral')
        batch_id: The ID of the batch to retrieve
        api_key: API key for the provider
        api_base: Base URL for the provider API
        client_args: Additional provider-specific arguments for client instantiation
        **kwargs: Additional provider-specific arguments

    Returns:
        The batch object

    """
    llm = AnyLLM.create(LLMProvider.from_string(provider), api_key=api_key, api_base=api_base, **client_args or {})
    return await llm.aretrieve_batch(batch_id, **kwargs)


@experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
def cancel_batch(
    provider: str | LLMProvider,
    batch_id: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Batch:
    """Cancel a batch job.

    Args:
        provider: Provider name to use for the request (e.g., 'openai', 'mistral')
        batch_id: The ID of the batch to cancel
        api_key: API key for the provider
        api_base: Base URL for the provider API
        client_args: Additional provider-specific arguments for client instantiation
        **kwargs: Additional provider-specific arguments

    Returns:
        The cancelled batch object

    """
    llm = AnyLLM.create(LLMProvider.from_string(provider), api_key=api_key, api_base=api_base, **client_args or {})
    return llm.cancel_batch(batch_id, **kwargs)


@experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
async def acancel_batch(
    provider: str | LLMProvider,
    batch_id: str,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Batch:
    """Cancel a batch job asynchronously.

    Args:
        provider: Provider name to use for the request (e.g., 'openai', 'mistral')
        batch_id: The ID of the batch to cancel
        api_key: API key for the provider
        api_base: Base URL for the provider API
        client_args: Additional provider-specific arguments for client instantiation
        **kwargs: Additional provider-specific arguments

    Returns:
        The cancelled batch object

    """
    llm = AnyLLM.create(LLMProvider.from_string(provider), api_key=api_key, api_base=api_base, **client_args or {})
    return await llm.acancel_batch(batch_id, **kwargs)


@experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
def list_batches(
    provider: str | LLMProvider,
    *,
    after: str | None = None,
    limit: int | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Sequence[Batch]:
    """List batch jobs.

    Args:
        provider: Provider name to use for the request (e.g., 'openai', 'mistral')
        after: A cursor for pagination. Returns batches after this batch ID.
        limit: Maximum number of batches to return (default: 20)
        api_key: API key for the provider
        api_base: Base URL for the provider API
        client_args: Additional provider-specific arguments for client instantiation
        **kwargs: Additional provider-specific arguments

    Returns:
        A list of batch objects

    """
    llm = AnyLLM.create(LLMProvider.from_string(provider), api_key=api_key, api_base=api_base, **client_args or {})
    return llm.list_batches(after=after, limit=limit, **kwargs)


@experimental(BATCH_API_EXPERIMENTAL_MESSAGE)
async def alist_batches(
    provider: str | LLMProvider,
    *,
    after: str | None = None,
    limit: int | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    client_args: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Sequence[Batch]:
    """List batch jobs asynchronously.

    Args:
        provider: Provider name to use for the request (e.g., 'openai', 'mistral')
        after: A cursor for pagination. Returns batches after this batch ID.
        limit: Maximum number of batches to return (default: 20)
        api_key: API key for the provider
        api_base: Base URL for the provider API
        client_args: Additional provider-specific arguments for client instantiation
        **kwargs: Additional provider-specific arguments

    Returns:
        A list of batch objects

    """
    llm = AnyLLM.create(LLMProvider.from_string(provider), api_key=api_key, api_base=api_base, **client_args or {})
    return await llm.alist_batches(after=after, limit=limit, **kwargs)
