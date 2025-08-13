from collections.abc import Callable, Iterator
from typing import Any

from pydantic import BaseModel

from any_llm.provider import ApiConfig, Provider, ProviderFactory
from any_llm.tools import prepare_tools
from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CreateEmbeddingResponse
from any_llm.types.responses import Response, ResponseInputParam, ResponseStreamEvent


def _prepare_completion_request(
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any] | Callable[..., Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_turns: int | None = None,
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
    api_timeout: float | None = None,
    user: str | None = None,
    **kwargs: Any,
) -> tuple[Provider, str, dict[str, Any]]:
    """Prepare a completion request by validating inputs and creating provider instance.

    Returns:
        tuple: (provider_instance, model_name, completion_kwargs)

    """
    provider_key, model_name = ProviderFactory.split_model_provider(model)

    config: dict[str, str] = {}
    if api_key:
        config["api_key"] = str(api_key)
    if api_base:
        config["api_base"] = str(api_base)
    api_config = ApiConfig(**config)

    provider = ProviderFactory.create_provider(provider_key, api_config)

    completion_kwargs = kwargs.copy()
    if tools is not None:
        completion_kwargs["tools"] = prepare_tools(tools)
    if tool_choice is not None:
        completion_kwargs["tool_choice"] = tool_choice
    if max_turns is not None:
        completion_kwargs["max_turns"] = max_turns
    if temperature is not None:
        completion_kwargs["temperature"] = temperature
    if top_p is not None:
        completion_kwargs["top_p"] = top_p
    if max_tokens is not None:
        completion_kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        completion_kwargs["response_format"] = response_format
    if stream is not None:
        completion_kwargs["stream"] = stream
    if n is not None:
        completion_kwargs["n"] = n
    if stop is not None:
        completion_kwargs["stop"] = stop
    if presence_penalty is not None:
        completion_kwargs["presence_penalty"] = presence_penalty
    if frequency_penalty is not None:
        completion_kwargs["frequency_penalty"] = frequency_penalty
    if seed is not None:
        completion_kwargs["seed"] = seed
    if api_timeout is not None:
        completion_kwargs["timeout"] = api_timeout
    if user is not None:
        completion_kwargs["user"] = user

    return provider, model_name, completion_kwargs


def completion(
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any] | Callable[..., Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_turns: int | None = None,
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
    api_timeout: float | None = None,
    user: str | None = None,
    **kwargs: Any,
) -> ChatCompletion | Iterator[ChatCompletionChunk]:
    """Create a chat completion.

    Args:
        model: Model identifier in format 'provider/model' (e.g., 'mistral/mistral-small')
        messages: List of messages for the conversation
        tools: List of tools for tool calling. Can be Python callables or OpenAI tool format dicts
        tool_choice: Controls which tools the model can call
        max_turns: Maximum number of tool execution turns
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
        api_timeout: Request timeout in seconds
        user: Unique identifier for the end user
        **kwargs: Additional provider-specific parameters

    Returns:
        The completion response from the provider

    """
    provider, model_name, completion_kwargs = _prepare_completion_request(
        model,
        messages,
        tools=tools,
        tool_choice=tool_choice,
        max_turns=max_turns,
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
        api_key=api_key,
        api_base=api_base,
        api_timeout=api_timeout,
        user=user,
        **kwargs,
    )

    return provider.completion(model_name, messages, **completion_kwargs)


async def acompletion(
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any] | Callable[..., Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_turns: int | None = None,
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
    api_timeout: float | None = None,
    user: str | None = None,
    **kwargs: Any,
) -> ChatCompletion | Iterator[ChatCompletionChunk]:
    """Create a chat completion asynchronously.

    Args:
        model: Model identifier in format 'provider/model' (e.g., 'mistral/mistral-small')
        messages: List of messages for the conversation
        tools: List of tools for tool calling. Can be Python callables or OpenAI tool format dicts
        tool_choice: Controls which tools the model can call
        max_turns: Maximum number of tool execution turns
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
        api_timeout: Request timeout in seconds
        user: Unique identifier for the end user
        **kwargs: Additional provider-specific parameters

    Returns:
        The completion response from the provider

    """
    provider, model_name, completion_kwargs = _prepare_completion_request(
        model,
        messages,
        tools=tools,
        tool_choice=tool_choice,
        max_turns=max_turns,
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
        api_key=api_key,
        api_base=api_base,
        api_timeout=api_timeout,
        user=user,
        **kwargs,
    )

    return await provider.acompletion(model_name, messages, **completion_kwargs)


def responses(
    model: str,
    input_data: str | ResponseInputParam,
    *,
    tools: list[dict[str, Any] | Callable[..., Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    api_timeout: float | None = None,
    user: str | None = None,
    **kwargs: Any,
) -> Response | Iterator[ResponseStreamEvent]:
    """Create a response using the OpenAI-style Responses API.

    This follows the OpenAI Responses API shape and returns the aliased
    `any_llm.types.responses.Response` type. If `stream=True`, an iterator of
    `any_llm.types.responses.ResponseStreamEvent` items is returned.

    Args:
        model: Model identifier in format 'provider/model' (e.g., 'openai/gpt-4o')
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
        api_timeout: Request timeout in seconds
        user: Unique identifier for the end user
        **kwargs: Additional provider-specific parameters

    Returns:
        Either a `Response` object (non-streaming) or an iterator of
        `ResponseStreamEvent` (streaming).

    Raises:
        NotImplementedError: If the selected provider does not support the Responses API.

    """
    provider_key, model_name = ProviderFactory.split_model_provider(model)

    config: dict[str, str] = {}
    if api_key:
        config["api_key"] = str(api_key)
    if api_base:
        config["api_base"] = str(api_base)
    api_config = ApiConfig(**config)

    provider = ProviderFactory.create_provider(provider_key, api_config)

    responses_kwargs = kwargs.copy()
    if tools is not None:
        responses_kwargs["tools"] = prepare_tools(tools)
    if tool_choice is not None:
        responses_kwargs["tool_choice"] = tool_choice
    if max_output_tokens is not None:
        responses_kwargs["max_output_tokens"] = max_output_tokens
    if temperature is not None:
        responses_kwargs["temperature"] = temperature
    if top_p is not None:
        responses_kwargs["top_p"] = top_p
    if stream is not None:
        responses_kwargs["stream"] = stream
    if api_timeout is not None:
        responses_kwargs["timeout"] = api_timeout
    if user is not None:
        responses_kwargs["user"] = user

    return provider.responses(model_name, input_data, **responses_kwargs)


async def aresponses(
    model: str,
    input_data: str | ResponseInputParam,
    *,
    tools: list[dict[str, Any] | Callable[..., Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stream: bool | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    api_timeout: float | None = None,
    user: str | None = None,
    **kwargs: Any,
) -> Response | Iterator[ResponseStreamEvent]:
    """Create a response using the OpenAI-style Responses API.

    This follows the OpenAI Responses API shape and returns the aliased
    `any_llm.types.responses.Response` type. If `stream=True`, an iterator of
    `any_llm.types.responses.ResponseStreamEvent` items is returned.

    Args:
        model: Model identifier in format 'provider/model' (e.g., 'openai/gpt-4o')
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
        api_timeout: Request timeout in seconds
        user: Unique identifier for the end user
        **kwargs: Additional provider-specific parameters

    Returns:
        Either a `Response` object (non-streaming) or an iterator of
        `ResponseStreamEvent` (streaming).

    Raises:
        NotImplementedError: If the selected provider does not support the Responses API.

    """
    provider_key, model_name = ProviderFactory.split_model_provider(model)

    config: dict[str, str] = {}
    if api_key:
        config["api_key"] = str(api_key)
    if api_base:
        config["api_base"] = str(api_base)
    api_config = ApiConfig(**config)

    provider = ProviderFactory.create_provider(provider_key, api_config)

    responses_kwargs = kwargs.copy()
    if tools is not None:
        responses_kwargs["tools"] = prepare_tools(tools)
    if tool_choice is not None:
        responses_kwargs["tool_choice"] = tool_choice
    if max_output_tokens is not None:
        responses_kwargs["max_output_tokens"] = max_output_tokens
    if temperature is not None:
        responses_kwargs["temperature"] = temperature
    if top_p is not None:
        responses_kwargs["top_p"] = top_p
    if stream is not None:
        responses_kwargs["stream"] = stream
    if api_timeout is not None:
        responses_kwargs["timeout"] = api_timeout
    if user is not None:
        responses_kwargs["user"] = user

    return await provider.aresponses(model_name, input_data, **responses_kwargs)


def embedding(
    model: str,
    inputs: str | list[str],
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    **kwargs: Any,
) -> CreateEmbeddingResponse:
    """Create an embedding.

    Args:
        model: Model identifier in format 'provider/model' (e.g., 'mistral/mistral-small')
        inputs: The input text to embed
        api_key: API key for the provider
        api_base: Base URL for the provider API
        **kwargs: Additional provider-specific parameters

    Returns:
        The embedding of the input text

    """
    provider_key, model_name = ProviderFactory.split_model_provider(model)

    config: dict[str, str] = {}
    if api_key:
        config["api_key"] = str(api_key)
    if api_base:
        config["api_base"] = str(api_base)
    api_config = ApiConfig(**config)

    provider = ProviderFactory.create_provider(provider_key, api_config)

    return provider.embedding(model_name, inputs, **kwargs)


async def aembedding(
    model: str,
    inputs: str | list[str],
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    **kwargs: Any,
) -> CreateEmbeddingResponse:
    """Create an embedding asynchronously.

    Args:
        model: Model identifier in format 'provider/model' (e.g., 'openai/text-embedding-3-small')
        inputs: The input text to embed
        api_key: API key for the provider
        api_base: Base URL for the provider API
        **kwargs: Additional provider-specific parameters

    Returns:
        The embedding of the input text

    """
    provider_key, model_name = ProviderFactory.split_model_provider(model)

    config: dict[str, str] = {}
    if api_key:
        config["api_key"] = str(api_key)
    if api_base:
        config["api_base"] = str(api_base)
    api_config = ApiConfig(**config)

    provider = ProviderFactory.create_provider(provider_key, api_config)

    return await provider.aembedding(model_name, inputs, **kwargs)
