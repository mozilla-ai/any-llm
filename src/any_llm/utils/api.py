from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from any_llm.provider import ApiConfig, Provider, ProviderFactory
from any_llm.tools import prepare_tools
from any_llm.types.completion import CompletionParams


def _process_completion_params(
    model: str,
    messages: list[dict[str, Any]],
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
    **kwargs: Any,
) -> tuple[Provider, CompletionParams]:
    provider_key, model_name = ProviderFactory.split_model_provider(model)

    config: dict[str, str] = {}
    if api_key:
        config["api_key"] = str(api_key)
    if api_base:
        config["api_base"] = str(api_base)
    api_config = ApiConfig(**config)

    provider = ProviderFactory.create_provider(provider_key, api_config)

    prepared_tools: list[dict[str, Any]] | None = None
    if tools:
        prepared_tools = prepare_tools(tools)

    completion_params = CompletionParams(
        model_id=model_name,
        messages=messages,
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
        timeout=api_timeout or kwargs.get("timeout", None),
    )
    return provider, completion_params
