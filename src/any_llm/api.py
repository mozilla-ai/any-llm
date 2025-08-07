from typing import Any, Optional, List, Union, Callable, Iterator

from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CreateEmbeddingResponse
from pydantic import BaseModel
from any_llm.provider import ProviderFactory, ApiConfig, Provider
from any_llm.tools import prepare_tools


def _prepare_completion_request(
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: Optional[List[Union[dict[str, Any], Callable[..., Any]]]] = None,
    tool_choice: Optional[Union[str, dict[str, Any]]] = None,
    max_turns: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: dict[str, Any] | type[BaseModel] | None = None,
    stream: Optional[bool] = None,
    n: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[Union[float, int]] = None,
    user: Optional[str] = None,
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
    if timeout is not None:
        completion_kwargs["timeout"] = timeout
    if user is not None:
        completion_kwargs["user"] = user

    return provider, model_name, completion_kwargs


def completion(
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: Optional[List[Union[dict[str, Any], Callable[..., Any]]]] = None,
    tool_choice: Optional[Union[str, dict[str, Any]]] = None,
    max_turns: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: dict[str, Any] | type[BaseModel] | None = None,
    stream: Optional[bool] = None,
    n: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[Union[float, int]] = None,
    user: Optional[str] = None,
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
        timeout: Request timeout in seconds
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
        timeout=timeout,
        user=user,
        **kwargs,
    )

    return provider.completion(model_name, messages, **completion_kwargs)


async def acompletion(
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: Optional[List[Union[dict[str, Any], Callable[..., Any]]]] = None,
    tool_choice: Optional[Union[str, dict[str, Any]]] = None,
    max_turns: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: dict[str, Any] | type[BaseModel] | None = None,
    stream: Optional[bool] = None,
    n: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    seed: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[Union[float, int]] = None,
    user: Optional[str] = None,
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
        timeout: Request timeout in seconds
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
        timeout=timeout,
        user=user,
        **kwargs,
    )

    return await provider.acompletion(model_name, messages, **completion_kwargs)


def verify_kwargs(provider_name: str, **kwargs: Any) -> None:
    """Check if the provider supports the kwargs.

    For example, if the provider does not yet support streaming, it will raise an error.

    This does not verify that the provider supports the model or that the model supports the kwargs.
    In order to determine that info you will need to refer to the provider documentation.

    Args:
        provider_name: The name of the provider to check
        **kwargs: The kwargs to check

    Returns:
        None

    Raises:
        UnsupportedParameterError: If the provider does not support the kwargs.
    """
    provider_key = ProviderFactory.get_provider_enum(provider_name)
    provider_cls = ProviderFactory.get_provider_class(provider_key)
    provider_cls.verify_kwargs(kwargs)


def embedding(
    model: str,
    inputs: str | list[str],
    *,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
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
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
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
