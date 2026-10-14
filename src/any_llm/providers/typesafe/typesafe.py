from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypeVar, overload

from pydantic import BaseModel, ValidationError
from typing_extensions import override

from any_llm.any_llm import AnyLLM
from any_llm.constants import INSIDE_NOTEBOOK
from any_llm.exceptions import AnyLLMError, InvalidRequestError, MissingApiKeyError, ProviderError
from any_llm.types.model import Model
from any_llm.utils.aio import run_async_in_sync
from any_llm.utils.exception_handler import handle_exceptions

MISSING_PACKAGES_ERROR: ImportError | None = None
try:
    from typesafe_sdk import (
        AsyncTypeSafeClient,
        ListModelsResponse,
        RetryPolicy,
        SystemOneResponse,
        TypeSafeAPIResponseValidationError,
        TypeSafeBadRequestError,
        TypeSafeError,
        TypeSafeUnprocessableEntityError,
    )
except ImportError as e:
    MISSING_PACKAGES_ERROR = e

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typesafe_sdk import JSONContent, Question

    from any_llm.types.completion import ChatCompletion, ChatCompletionChunk, CompletionParams, CreateEmbeddingResponse

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class TypesafeProvider(AnyLLM):
    """Provider for TypeSafe AI.

    TypeSafe exposes System One, which answers named questions (noul, choice, score) about a
    state. It has no chat completions, so use `adecide` instead of `completion`.

    A base URL other than the hosted API (for example a local Ollama server) makes the API key
    optional. `list_models` only works against the hosted API, whose response shape the SDK expects.
    """

    PROVIDER_NAME = "typesafe"
    ENV_API_KEY_NAME = "TYPESAFE_API_KEY"
    ENV_API_BASE_NAME = "TYPESAFE_BASE_URL"
    API_BASE = "https://api.typesafe.ai"
    PROVIDER_DOCUMENTATION_URL = "https://docs.typesafe.ai/sdk/python"

    SUPPORTS_COMPLETION = False
    SUPPORTS_COMPLETION_STREAMING = False
    SUPPORTS_COMPLETION_REASONING = False
    SUPPORTS_COMPLETION_IMAGE = False
    SUPPORTS_COMPLETION_PDF = False
    SUPPORTS_MESSAGES = False
    SUPPORTS_RESPONSES = False
    SUPPORTS_EMBEDDING = False
    SUPPORTS_LIST_MODELS = True
    SUPPORTS_BATCH = False
    MAX_RETRIES_SUPPORT = "mapped"

    MISSING_PACKAGES_ERROR = MISSING_PACKAGES_ERROR

    client: AsyncTypeSafeClient

    @override
    def __init__(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        # AnyLLM.__init__ verifies the key before it resolves the base URL, and the key rule depends on it.
        resolved_api_base = self._resolve_api_base(api_base)
        self._self_hosted = resolved_api_base is not None and resolved_api_base.rstrip("/") != self.API_BASE
        super().__init__(api_key=api_key, api_base=api_base, **kwargs)

    @override
    def _resolve_api_base(self, api_base: str | None = None) -> str | None:
        # A blank value (an empty TYPESAFE_BASE_URL= line) means "unset", as it does for the SDK itself.
        return (super()._resolve_api_base(api_base) or "").strip() or None

    @override
    def _verify_and_set_api_key(self, api_key: str | None = None) -> str | None:
        # The SDK strips the key, so a whitespace-only key counts as missing here too.
        api_key = (api_key or os.getenv(self.ENV_API_KEY_NAME) or "").strip()
        if api_key:
            return api_key
        if self._self_hosted:
            # A self-hosted server (for example Ollama) needs no key, but the SDK client still requires one.
            return "no-key-required"
        raise MissingApiKeyError(self.PROVIDER_NAME, self.ENV_API_KEY_NAME)

    @staticmethod
    @override
    def _convert_completion_params(params: CompletionParams, **kwargs: Any) -> dict[str, Any]:
        msg = "TypeSafe does not support completions"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_completion_response(response: Any) -> ChatCompletion:
        msg = "TypeSafe does not support completions"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_completion_chunk_response(response: Any, **kwargs: Any) -> ChatCompletionChunk:
        msg = "TypeSafe does not support completions"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_embedding_params(params: Any, **kwargs: Any) -> dict[str, Any]:
        msg = "TypeSafe does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_embedding_response(response: Any) -> CreateEmbeddingResponse:
        msg = "TypeSafe does not support embeddings"
        raise NotImplementedError(msg)

    @staticmethod
    @override
    def _convert_list_models_response(response: ListModelsResponse) -> Sequence[Model]:
        models = []
        for model in response.models:
            # The SDK types release_date as a plain string, so a malformed date gets this far.
            try:
                released = datetime.fromisoformat(model.release_date)
            except ValueError as error:
                msg = f"TypeSafe returned an invalid release_date {model.release_date!r} for model {model.name!r}"
                raise ProviderError(msg, provider_name=TypesafeProvider.PROVIDER_NAME) from error
            models.append(
                Model(
                    id=model.name,
                    created=int(released.replace(tzinfo=UTC).timestamp()),
                    object="model",
                    owned_by=TypesafeProvider.PROVIDER_NAME,
                )
            )
        return models

    @staticmethod
    @override
    def _classify_exception(exception: Exception) -> type[AnyLLMError] | None:
        if isinstance(exception, TypeSafeAPIResponseValidationError):
            # The SDK raises this both for a 200 that fails its own response model (a server fault) and
            # for one that fails the caller's `response_model` (a mismatch no retry can fix). The pydantic
            # error it chains names the model that failed.
            cause = exception.__cause__
            if isinstance(cause, ValidationError) and cause.title not in {
                SystemOneResponse.__name__,
                ListModelsResponse.__name__,
            }:
                return InvalidRequestError
            return ProviderError
        if isinstance(exception, (TypeSafeBadRequestError, TypeSafeUnprocessableEntityError)):
            # The message lists the caller's question names, which must not pick the class.
            return InvalidRequestError
        if type(exception) is TypeSafeError:
            # The SDK raises the base class for caller mistakes it catches before sending anything.
            return InvalidRequestError
        return None

    @override
    def _apply_max_retries(self, max_retries: int, client_kwargs: dict[str, Any]) -> None:
        if client_kwargs.get("retry") is None:
            # The SDK's default 30 s retry budget would otherwise cut the retry count short.
            client_kwargs["retry"] = RetryPolicy(max_retries=max_retries, timeout=None)

    @override
    def _init_client(self, api_key: str | None = None, api_base: str | None = None, **kwargs: Any) -> None:
        self.client = AsyncTypeSafeClient(api_key=api_key, base_url=api_base, **kwargs)

    @override
    async def _alist_models(self, **kwargs: Any) -> Sequence[Model]:
        return self._convert_list_models_response(await self.client.models.list(**kwargs))

    @overload
    def decide(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        response_model: None = None,
        **kwargs: Any,
    ) -> SystemOneResponse: ...

    @overload
    def decide(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        response_model: type[ResponseT],
        **kwargs: Any,
    ) -> ResponseT: ...

    def decide(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        response_model: type[ResponseT] | None = None,
        **kwargs: Any,
    ) -> SystemOneResponse | ResponseT:
        """Synchronous version of `adecide`."""
        allow_running_loop = kwargs.pop("allow_running_loop", INSIDE_NOTEBOOK)
        kwargs["response_model"] = response_model
        result: SystemOneResponse | ResponseT = run_async_in_sync(
            self.adecide(state, questions, model=model, **kwargs), allow_running_loop=allow_running_loop
        )
        return result

    @overload
    async def adecide(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        response_model: None = None,
        **kwargs: Any,
    ) -> SystemOneResponse: ...

    @overload
    async def adecide(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        response_model: type[ResponseT],
        **kwargs: Any,
    ) -> ResponseT: ...

    @handle_exceptions()
    async def adecide(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        response_model: type[ResponseT] | None = None,
        **kwargs: Any,
    ) -> SystemOneResponse | ResponseT:
        """Answer named questions about a state with TypeSafe System One.

        Args:
            state: Text, a JSON object, or an array to evaluate.
            questions: Names mapped to `Noul`, `Choice` or `Score` objects (or their dict forms).
            model: Model override. `None` uses the client default: `TYPESAFE_DEFAULT_MODEL`, else the
                hosted alias `jev-latest`, which a self-hosted server usually does not serve.
            response_model: Optional Pydantic model to parse the answers into instead of `SystemOneResponse`.
            **kwargs: Passed to `AsyncTypeSafeClient.system_one` (for example `timeout`).

        Returns:
            An instance of `response_model`, or the SDK's `SystemOneResponse` with answers grouped by
            `nouls`, `choices` and `scores`.
        """
        result: SystemOneResponse | ResponseT = await self.client.system_one(
            state, questions, model=model, response_model=response_model, **kwargs
        )
        return result
