from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from any_llm.any_llm import AnyLLM
    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk


class AnyAPI:
    def __init__(self, any_api_key: str, provider_class: type[AnyLLM], api_base: str | None = None, **kwargs: Any) -> None:
        self.any_api_key = any_api_key
        self.provider_class = provider_class
        self.api_base = api_base
        self.kwargs = kwargs

        self.provider_instance: AnyLLM | None = None
        self.api_key = None

    def _init_provider(self) -> None:
        api_key = self.get_provider_key()
        self.provider_instance = self.provider_class(api_key=api_key, api_base=self.api_base, **self.kwargs)

    def get_provider_key(self) -> str:
        raise NotImplementedError

    def completion(
        self,
        **kwargs: Any,
    ) -> ChatCompletion | Iterator[ChatCompletionChunk]:
        if self.provider_instance is None:
            self._init_provider()
        result = self.provider_instance.completion(**kwargs)
        self.post_metadata(result)
        return result

    def post_metadata(self, result):
        raise NotImplementedError
