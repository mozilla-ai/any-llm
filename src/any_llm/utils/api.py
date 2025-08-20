from collections.abc import Callable
from typing import Any, Literal, cast

from pydantic import BaseModel

from any_llm.provider import ApiConfig, Provider, ProviderFactory, ProviderName
from any_llm.tools import prepare_tools
from any_llm.types.completion import ChatCompletionMessage, CompletionParams


def _process_completion_params(
    model: str,
    provider: str | ProviderName | None,
    messages: list[dict[str, Any] | ChatCompletionMessage],
    tools: list[dict[str, Any] | Callable[..., Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    response_format: dict[str, Any] | type[BaseModel] | None,
    stream: bool | None,
    n: int | None,
    stop: str | list[str] | None,
    presence_penalty: float | None,
    frequency_penalty: float | None,
    seed: int | None,
    api_key: str | None,
    api_base: str | None,
    api_timeout: float | None,
    user: str | None,
    parallel_tool_calls: bool | None,
    logprobs: bool | None,
    top_logprobs: int | None,
    logit_bias: dict[str, float] | None,
    stream_options: dict[str, Any] | None,
    max_completion_tokens: int | None,
    reasoning_effort: Literal["minimal", "low", "medium", "high", "auto"] | None,
    **kwargs: Any,
) -> tuple[Provider, CompletionParams]:
    if provider is None:
        provider_key, model_name = ProviderFactory.split_model_provider(model)
    else:
        provider_key = ProviderName.from_string(provider)
        model_name = model

    config: dict[str, str] = {}
    if api_key:
        config["api_key"] = str(api_key)
    if api_base:
        config["api_base"] = str(api_base)
    api_config = ApiConfig(**config)

    provider_instance = ProviderFactory.create_provider(provider_key, api_config)

    prepared_tools: list[dict[str, Any]] | None = None
    if tools:
        prepared_tools = prepare_tools(tools)

    for i, message in enumerate(messages):
        if isinstance(message, ChatCompletionMessage):
            # Dump the message but exclude the extra field that we extend from OpenAI Spec
            messages[i] = message.model_dump(exclude_none=True, exclude={"reasoning"})

    completion_params = CompletionParams(
        model_id=model_name,
        messages=cast("list[dict[str, Any]]", messages),
        tools=prepared_tools,
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
    )
    return provider_instance, completion_params
